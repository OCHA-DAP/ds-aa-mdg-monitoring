from typing import Literal, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
)
from shapely.affinity import translate
from shapely.geometry import Point, Polygon, shape
from tqdm.auto import tqdm

from src.constants import NAUTICAL_MILE_TO_KM
from src.utils.exposure import calculate_single_adm_exposure

SECTOR_ORDER = ("NEQ", "NWQ", "SEQ", "SWQ")
QUADS = ["ne", "se", "sw", "nw"]


def _nautical_mile_to_km(nautical_miles):
    """Convert nautical miles to kilometers."""
    return nautical_miles * NAUTICAL_MILE_TO_KM


def read_uncertainty_cone(track_json):
    """Read the uncertainty cone from the track JSON."""
    uncertainty_cone = None
    for feature in track_json.get("features", []):
        properties = feature.get("properties", {})
        if properties.get("data_type") == "uncertainty_cone":
            geometry = feature.get("geometry", {})
            uncertainty_cone = shape(geometry)
            break
    return uncertainty_cone


def read_forecast_details(track_json):
    """Read forecast details like cyclone name, season, reference time, and basin."""
    fc_details = {
        "cyclone_name": track_json.get("cyclone_name"),
        "season": track_json.get("season"),
        "reference_time": track_json.get("reference_time"),
        "basin": track_json.get("basin"),
    }
    return fc_details


def read_records(track_json):
    """
    Read cyclone track records from a JSON file and convert them into a GeoDataFrame.

    Args:
        track_json (dict): JSON object containing cyclone track data, typically loaded from a GeoJSON file.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the parsed cyclone track data with geometry and wind contours.
    """
    records = []

    # Loop through each feature in the JSON file
    for feature in track_json.get("features", []):
        # Extract properties and geometry from the feature
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")  # 'Point' or 'Polygon'
        geometry_coords = geometry.get(
            "coordinates", []
        )  # Coordinates of the geometry

        # Create a dictionary to store the record's attributes
        record = {
            "data_type": properties.get(
                "data_type"
            ),  # Type of data (e.g., observed, forecasted)
            "time": pd.Timestamp(properties.get("time")),  # Time of the record
            "position_accuracy": properties.get(
                "position_accuracy"
            ),  # Accuracy of the recorded position
            "development": properties.get("cyclone_data", {}).get(
                "development"
            ),  # Development stage of the cyclone
            "maximum_wind_speed": _nautical_mile_to_km(
                float(
                    properties.get("cyclone_data", {})
                    .get("maximum_wind", {})
                    .get("wind_speed_kt", 0)
                )
            ),  # Maximum sustained wind speed (converted to km/h)
            "maximum_wind_gust": _nautical_mile_to_km(
                float(
                    properties.get("cyclone_data", {})
                    .get("maximum_wind", {})
                    .get("wind_speed_gust_kt", 0)
                )
            ),  # Maximum wind gust (converted to km/h)
            "geometry": (
                Point(*geometry_coords)
                if geometry_type == "Point"
                else Polygon(*geometry_coords)
            ),  # Create geometry
        }

        # Extract wind contour data
        wind_contours = properties.get("cyclone_data", {}).get(
            "wind_contours", []
        )
        for contour in wind_contours:
            # Wind speed for this contour
            wind_speed_kt = contour.get("wind_speed_kt")

            # Extract radii for different sectors and convert to km
            radii = {
                sector_data.get("sector"): _nautical_mile_to_km(
                    sector_data.get("value")
                )
                for sector_data in contour.get("radius", [])
                if sector_data.get("value")
                is not None  # Skip if no value is provided
            }

            # Add radii data to the record with a key based on the wind speed
            record[f"wind_contour_{wind_speed_kt}kt"] = radii

        # Append the record to the list if the record is not the uncertainty cone
        if record["data_type"] in ["analysis", "forecast"]:
            records.append(record)

    # Convert the list of records into a GeoDataFrame
    records = gpd.GeoDataFrame(records)
    return records


def parse_track_json(track_json):
    """
    Parse the cyclone track data from a JSON file.

    Args:
        track_json (dict): JSON data containing cyclone track information.

    Returns:
        tuple: A tuple containing the parsed DataFrame, cyclone details, and exposed areas set.
    """
    uncertainty_cone = read_uncertainty_cone(track_json)
    fc_details = read_forecast_details(track_json)
    records = read_records(track_json)

    return records, fc_details, uncertainty_cone


def radii_dict_to_list(r):
    if not isinstance(r, dict):
        return [np.nan, np.nan, np.nan, np.nan]
    return [r.get(s, np.nan) for s in SECTOR_ORDER]


def prepare_wind_contours(df):
    df = df.copy()

    contour_cols = [c for c in df.columns if c.startswith("wind_contour_")]

    for col in contour_cols:
        # only convert if still dicts
        if isinstance(df[col].dropna().iloc[0], dict):
            df[col] = df[col].apply(radii_dict_to_list)

    return df


def expand_quad_col(df, col):
    if f"{col}_ne" in df:
        print(f"already done for {col}")
        return df
    df_expanded = (
        df[col]
        .apply(pd.Series)
        .rename(
            columns={
                0: f"{col}_ne",
                1: f"{col}_nw",
                2: f"{col}_se",
                3: f"{col}_sw",
            }
        )
    )
    return df.join(df_expanded)


def _radius_from_quadrants(
    theta_deg: np.ndarray, ne: float, se: float, sw: float, nw: float
) -> np.ndarray:
    """
    Return radius for each angle by linearly interpolating between the
    four quadrant control points defined at bearings:
        45°  -> NE
        135° -> NW
        225° -> SW
        315° -> SE
    Bearing convention: 0° = East, 90° = North (mathematical).
    """
    # Control bearings (deg) and radii, with wrap-around point to close the loop
    bearings = np.array([45, 135, 225, 315, 405], dtype=float)
    radii = np.array([ne, nw, sw, se, ne], dtype=float)

    # Map all thetas into [0, 360) and also allow values up to 405 for interpolation
    t = (theta_deg % 360).astype(float)
    # For values in [0,45), make an equivalent in [360,405) to interpolate to NE nicely
    t_wrap = t.copy()
    t_wrap[t < 45] += 360

    # Interpolate and then map back (the interpolation function is periodic due to control duplication)
    r = np.interp(t_wrap, bearings, radii)
    return r


def make_quadrant_disk(
    center_xy: Tuple[float, float],
    ne: float,
    se: float,
    sw: float,
    nw: float,
    n_points: int = 360,
) -> Polygon:
    """
    Build a smooth polygon around (x, y) using quadrant radii. Units assumed meters.
    - center_xy: (x, y) in EPSG:3832
    - ne, se, sw, nw: radii for quadrants (meters)
    - n_points: angular resolution
    Bearing convention: 0°=East, 90°=North; polygon traced counter-clockwise.
    """
    x0, y0 = center_xy
    theta = np.linspace(0, 360, n_points, endpoint=False)  # degrees
    r = _radius_from_quadrants(theta, ne, se, sw, nw)

    # Convert polar -> Cartesian
    th = np.deg2rad(theta)
    xs = x0 + r * np.cos(th)
    ys = y0 + r * np.sin(th)

    # Ensure valid ring: close the polygon
    coords = np.column_stack([xs, ys])
    return Polygon(coords)


def build_merged_wind_buffer(
    gdf: gpd.GeoDataFrame,
    quad_cols: Tuple[str, str, str, str],
):
    """
    Build a merged wind buffer polygon from quadrant radii columns.
    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with point geometries and quadrant radius columns
    quad_cols: Tuple[str, str, str, str]
        Names of the four quadrant radius columns in order:
        (ne_col, se_col, sw_col, nw_col)

    Returns
    -------
    gpd.GeoSeries or None
        Merged polygon of wind buffers, or None if all radius values are NaN

    """
    ne_col, se_col, sw_col, nw_col = quad_cols
    polys = []
    gdf[[ne_col, se_col, sw_col, nw_col]] = (
        gdf[[ne_col, se_col, sw_col, nw_col]].fillna(0)
        * NAUTICAL_MILE_TO_KM
        * 1000
    )
    for _, row in gdf.iterrows():
        if row[[ne_col, se_col, sw_col, nw_col]].isna().all():
            return None

        poly = make_quadrant_disk(
            (row.geometry.x, row.geometry.y),
            row[ne_col],
            row[se_col],
            row[sw_col],
            row[nw_col],
        )
        polys.append(poly)
    return gpd.GeoSeries(polys).union_all()


def calculate_wind_buffers_gdf(
    df: pd.DataFrame,
    quad_cols_format: str = "quadrant_radius_{speed}_{quad}",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    valid_time_col: str = "valid_time",
    speeds=None,
):
    """
    Calculate wind buffer polygons for given wind speed quadrants.
    Note that this function interpolates the storm track to a regular
    30-minute interval before calculating the wind buffers.
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with storm track data including quadrant radius columns
    quad_cols_format: str = 'quadrant_radius_{speed}_{quad}'
        Format string for quadrant radius columns, with placeholders for
        speed and quad (e.g., 'quadrant_radius_{speed}_{quad}')
    lon_col: str = 'Longitude'
        Name of the longitude column in df
    lat_col: str = 'Latitude'
        Name of the latitude column in df
    valid_time_col: str = 'valid_time'
        Name of the valid time column in df

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with wind buffer polygons for each speed

    """
    if speeds is None:
        speeds = [34, 50, 64]
    all_quad_cols = [
        quad_cols_format.format(speed=speed, quad=x)
        for speed in speeds
        for x in QUADS
    ]
    df = df[[lon_col, lat_col, valid_time_col] + all_quad_cols].copy()
    df[lon_col] = df[lon_col].apply(lambda x: (x + 360) % 360)
    df_interp = interpolate_track(
        df,
        time_col=valid_time_col,
        lat_col=lat_col,
        lon_col=lon_col,
        freq="30min",
    )
    gdf_interp = gpd.GeoDataFrame(
        data=df_interp,
        geometry=gpd.points_from_xy(df_interp[lon_col], df_interp[lat_col]),
        crs=4326,
    ).to_crs(3857)
    dicts = []
    geoms = []
    for speed in speeds:
        speed_quad_cols = tuple(
            quad_cols_format.format(speed=speed, quad=x) for x in QUADS
        )
        geoms.append(build_merged_wind_buffer(gdf_interp, speed_quad_cols))
        dicts.append({"speed": speed})
    return gpd.GeoDataFrame(dicts, geometry=geoms, crs=3857)


def interpolate_track(
    df: pd.DataFrame,
    time_col: str = "valid_time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    freq: str = "30min",
    method: Literal["pchip", "akima", "cubic", "linear"] = "pchip",
    include_ends: bool = True,
) -> pd.DataFrame:
    """
    Resample a (time, lat, lon, ...) track to a regular grid (default 30 min).
    - Lat/lon use chosen spline method (default 'pchip').
    - All other numeric columns are interpolated linearly.
    - Assumes longitude already in [0, 360) and keeps it in [0, 360).
    - No extrapolation beyond the observed time span.
    - If only one point is available, return that point (same output schema).
    """

    # --- Prep ---
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.sort_values(time_col).drop_duplicates(
        subset=[time_col], keep="first"
    )
    work = work.dropna(subset=[lat_col, lon_col])

    n = len(work)
    if n == 0:
        # Nothing usable
        return pd.DataFrame(columns=[time_col, lat_col, lon_col]).astype(
            {time_col: "datetime64[ns, UTC]"}
        )

    # If exactly one point, return it in the same format (reset index, include numeric cols)
    if n == 1:
        row = work.iloc[0]
        out = pd.DataFrame(
            {
                time_col: [row[time_col]],
                lat_col: [float(row[lat_col])],
                lon_col: [float(row[lon_col]) % 360.0],
            }
        )
        # carry other numeric columns as-is
        other_cols = work.select_dtypes(
            include=[np.number]
        ).columns.difference([lat_col, lon_col])
        for col in other_cols:
            out[col] = float(row[col])
        return out.reset_index(drop=True)

    # --- target time grid
    tmin, tmax = work[time_col].min(), work[time_col].max()
    start = tmin.floor(freq) if include_ends else tmin.ceil(freq)
    end = tmax.ceil(freq) if include_ends else tmax.floor(freq)
    target = pd.date_range(start, end, freq=freq, tz="UTC")
    target = target[(target >= tmin) & (target <= tmax)]
    if target.empty:
        target = pd.DatetimeIndex([tmin, tmax])

    # --- time axis
    t0 = work[time_col].iloc[0]
    x = (work[time_col] - t0).dt.total_seconds().to_numpy()
    x_new = (pd.Series(target) - t0).dt.total_seconds().to_numpy()

    # --- lat/lon interpolation ---
    y_lat = work[lat_col].to_numpy(float)
    y_lon = work[lon_col].to_numpy(float)

    if method == "linear" or (method in ("akima", "cubic") and n < 3):
        interp_lat = lambda xv: np.interp(xv, x, y_lat)
        interp_lon = lambda xv: np.mod(np.interp(xv, x, y_lon), 360.0)
    elif method == "pchip":
        interp_lat = PchipInterpolator(x, y_lat)
        interp_lon = lambda xv: np.mod(PchipInterpolator(x, y_lon)(xv), 360.0)
    elif method == "akima":
        interp_lat = Akima1DInterpolator(x, y_lat)
        interp_lon = lambda xv: np.mod(
            Akima1DInterpolator(x, y_lon)(xv), 360.0
        )
    elif method == "cubic":
        interp_lat = CubicSpline(x, y_lat, bc_type="natural")
        interp_lon = lambda xv: np.mod(
            CubicSpline(x, y_lon, bc_type="natural")(xv), 360.0
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    lat_new = interp_lat(x_new)
    lon_new = interp_lon(x_new)

    # --- other numeric columns (linear only) ---
    other_cols = work.select_dtypes(include=[np.number]).columns.difference(
        [lat_col, lon_col]
    )
    out = pd.DataFrame(index=target)
    out[lat_col] = lat_new
    out[lon_col] = lon_new
    for col in other_cols:
        y = work[col].to_numpy(float)
        out[col] = np.interp(x_new, x, y)

    out.index.name = time_col
    out = out.reset_index()
    return out


def shift_gdf_points(
    gdf: gpd.GeoDataFrame,
    azimuth_deg: float,
    distance_col: str = "uncertainty_m",
    geographic_crs: str = "EPSG:4326",
    projected_crs: str = "EPSG:3857",
    longitude_col: str = "Longitude",
    latitude_col: str = "Latitude",
):
    gdf = gdf.to_crs(projected_crs)
    angle_rad = np.deg2rad(azimuth_deg)
    dx = gdf[distance_col] * np.sin(angle_rad)
    dy = gdf[distance_col] * np.cos(angle_rad)
    new_geometry = [
        translate(geom, xoff=x, yoff=y)
        for geom, x, y in zip(gdf.geometry, dx, dy)
    ]
    df_out = gdf.drop(columns="geometry")
    df_out["shift_deg"] = azimuth_deg
    df_out["shift_distance_m"] = gdf[distance_col]
    gdf_out = gpd.GeoDataFrame(df_out, geometry=new_geometry, crs=gdf.crs)
    gdf_out = gdf_out.to_crs(geographic_crs)
    gdf_out[longitude_col] = gdf_out.geometry.x
    gdf_out[latitude_col] = gdf_out.geometry.y
    return gdf_out


def calculate_shifted_exposures(
    gdf,
    da_wp: xr.DataArray,
    disable_tqdm=True,
    deg_step: int = 10,
    uncertainty_nm_col: str = "position_accuracy",
    time_col: str = "valid_time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    quad_cols_format: str = "quadrant_radius_{speed}_{quad}",
    speeds=None,
) -> (pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame):
    if speeds is None:
        speeds = [34, 50, 64]
    gdfs_shifts = []
    gdfs_shifts_buffers = []
    dfs = []
    gdf["uncertainty_m"] = gdf[uncertainty_nm_col] * NAUTICAL_MILE_TO_KM * 1000
    for shift_deg in tqdm(range(0, 360, deg_step), disable=disable_tqdm):
        _gdf_shift = shift_gdf_points(
            gdf, shift_deg, longitude_col=lon_col, latitude_col=lat_col
        )
        _gdf_shift_buffers = calculate_wind_buffers_gdf(
            _gdf_shift,
            lon_col=lon_col,
            lat_col=lat_col,
            valid_time_col=time_col,
            quad_cols_format=quad_cols_format,
            speeds=speeds,
        )
        _gdf_shift_buffers = _gdf_shift_buffers.to_crs(4326)
        _gdf_shift_buffers["shift_deg"] = shift_deg

        gdfs_shifts.append(_gdf_shift)
        gdfs_shifts_buffers.append(_gdf_shift_buffers)
        _df_exp = calculate_single_adm_exposure(_gdf_shift_buffers, da_wp)
        _df_exp["shift_deg"] = shift_deg
        dfs.append(_df_exp)
    gdf_shift_tracks = pd.concat(gdfs_shifts)
    gdf_shift_buffers = pd.concat(gdfs_shifts_buffers)
    df_exp_shift_raw = pd.concat(dfs, ignore_index=True)
    df_exp_shift = df_exp_shift_raw.pivot(
        columns="speed",
        index="shift_deg",
        values="pop_exposed",
    )
    df_exp_shift.columns = [f"exp_{x}" for x in df_exp_shift.columns]
    df_exp_shift = df_exp_shift.reset_index()
    return df_exp_shift, gdf_shift_buffers, gdf_shift_tracks
