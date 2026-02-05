import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, shape

from src.constants import NAUTICAL_MILE_TO_KM


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
