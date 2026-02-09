---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: ds-aa-mdg-monitoring
    language: python
    name: ds-aa-mdg-monitoring
---

# Wind exposure plot

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import geopandas as gpd
import ocha_stratus as stratus
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mtick

import numpy as np

from src.datasources.meteofr import (
    parse_track_json,
    prepare_wind_contours,
    expand_quad_col,
    calculate_wind_buffers_gdf,
    calculate_shifted_exposures,
)
from src.monitoring.plotting import plot_bullseye_exposures, plot_wind_buffers
from src.utils.exposure import calculate_multi_adm_exposure
from src.utils.blob_utils import PROJECT_PREFIX
```

## Load data

### CODAB

```python
adm1 = stratus.codab.load_codab_from_fieldmaps(iso3="mdg", admin_level=1)
```

```python
adm1.plot()
```

### Meteo France track forecast

```python
# from blob
blob_name = (
    "RSMC_LaReunion/CMRSTRACK_SWI$10_20252026_10-20252026_2026_02_07_12Z.json"
)
data = stratus.load_blob_data(blob_name, container_name="meteofr")

fcast_dict = json.loads(data)
```

```python
# local file
local_path = "temp/CMRSTRACK_SWI$10_20252026_GEZANI_2026_02_09_18Z.json"

with open(local_path, "r", encoding="utf-8") as f:
    fcast_dict = json.load(f)
```

```python
records, fc_details, uncertainty_cone = parse_track_json(fcast_dict)
```

```python
speeds = [28, 34, 48, 64]
```

```python
records = prepare_wind_contours(records)

for speed in speeds:
    col = f"wind_contour_{speed}kt"
    records = expand_quad_col(records, col)
```

```python
records["lon"] = records.geometry.x
records["lat"] = records.geometry.y
```

### WP

```python
blob_name = "worldpop/pop_count/global_pop_2026_CN_1km_R2025A_UA_v1.tif"
da_wp_global = stratus.open_blob_cog(blob_name, container_name="raster")
```

```python
da_wp = da_wp_global.rio.clip(adm1.geometry).squeeze(drop=True).compute()
```

```python
da_wp.attrs["_FillValue"] = None
da_wp = da_wp.where(da_wp > 0)
```

```python
da_wp.plot()
```

```python
da_wp.sum()
```

### Bubbles Template

```python
blob_name = f"{PROJECT_PREFIX}/processed/plotting/adm1_template.parquet"
template_df = stratus.load_parquet_from_blob(blob_name)
```

## Calculate exposure


### Calculate wind buffers

```python
gdf_buffers = calculate_wind_buffers_gdf(
    records,
    valid_time_col="time",
    quad_cols_format="wind_contour_{speed}kt_{quad}",
    lat_col="lat",
    lon_col="lon",
    speeds=speeds,
)
```

```python
gdf_buffers
```

```python
fig, ax = plt.subplots()
gdf_buffers.to_crs(4326).plot(ax=ax, alpha=0.3)
records.plot(ax=ax)
adm1.boundary.plot(ax=ax)
da_wp.plot(ax=ax, vmax=1000)
```

### Overlay and calculate exposure

```python
df_exp = calculate_multi_adm_exposure(
    gdf_buffers, da_wp, adm1, adm_index="adm1_src", disable_tqdm=False
)
```

```python
df_exp
```

```python
df_exp["speed_kmh"] = (df_exp["speed"] * 1.852).astype(int)
```

```python
df_exp["speed_kmh"].unique()
```

```python
df_exp
```

## Calculate perturbed exposures

```python
records = records.set_crs(4326)
```

```python
(
    df_exp_shift,
    gdf_shift_buffers,
    gdf_shift_tracks,
) = calculate_shifted_exposures(
    records,
    da_wp,
    disable_tqdm=False,
    time_col="time",
    lat_col="lat",
    lon_col="lon",
    quad_cols_format="wind_contour_{speed}kt_{quad}",
    speeds=speeds,
)
```

```python
df_exp_shift = df_exp_shift.sort_values(
    [f"exp_{x}" for x in sorted(speeds, reverse=True)], ascending=False
)
worst_row = df_exp_shift.iloc[0].copy()
worst_row["level"] = "worst"
best_row = df_exp_shift.iloc[-1].copy()
best_row["level"] = "best"
```

```python
df_exp_shift
```

```python
# Calculate best and worst case adm3 exposures
gdf_shift_buffers["speed_kmh"] = (gdf_shift_buffers["speed"] * 1.852).astype(
    int
)

worst_buffers = gdf_shift_buffers[
    gdf_shift_buffers["shift_deg"] == worst_row["shift_deg"]
]
best_buffers = gdf_shift_buffers[
    gdf_shift_buffers["shift_deg"] == best_row["shift_deg"]
]


df_exp_adm1_worst = calculate_multi_adm_exposure(
    worst_buffers, da_wp, adm1, adm_index="adm1_src", disable_tqdm=False
)

df_exp_adm1_best = calculate_multi_adm_exposure(
    best_buffers, da_wp, adm1, adm_index="adm1_src", disable_tqdm=False
)
```

## Plot

```python
colors = {51: "gold", 62: "darkorange", 88: "crimson", 118: "indigo"}
```

```python
colors.keys()
```

```python
fc_details
```

```python
def dt_to_EAT(s):
    dt_utc = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt_utc.astimezone(ZoneInfo("Africa/Nairobi"))
```

```python
dt_eat = dt_to_EAT(fc_details["reference_time"])
```

```python
issued_time_str = dt_eat.strftime("%Y-%m-%d %H:%M")
```

```python
gdf_buffers["speed_kmh"] = (gdf_buffers["speed"] * 1.852).astype(int)
```

```python
def plot_both_exp(
    gdf_tracks_plot, gdf_buffers_plot, df_exp_plot, forecast_label: str = ""
):
    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(12, 8),
        dpi=200,
    )

    fig_size = (9, 8)

    plot_wind_buffers(
        adm1,
        gdf_buffers_plot,
        colors=colors,
        speed_unit="km/h",
        speed_col="speed_kmh",
        ax=ax1,
        fig_size=fig_size,
        show_labels=True,
    )

    xs = gdf_tracks_plot.geometry.x.values
    ys = gdf_tracks_plot.geometry.y.values

    # line first (under points)
    ax1.plot(
        xs,
        ys,
        color="black",
        linewidth=1.5,
        zorder=9,
    )

    # points on top
    gdf_tracks_plot.plot(
        ax=ax1,
        color="black",
        markersize=20,
        zorder=10,
    )

    gpd.GeoSeries([uncertainty_cone], crs=adm1.crs).plot(
        ax=ax1,
        facecolor="none",
        edgecolor="grey",
        linewidth=1.5,
        linestyle="--",
        zorder=10,
    )

    existing_legend = ax1.get_legend()
    if existing_legend is not None:
        ax1.add_artist(existing_legend)

    # --- build proxy artists for track & cone ---
    track_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=1.5,
        label=forecast_label,
    )

    cone_handle = Patch(
        facecolor="none",
        edgecolor="grey",
        linewidth=1.5,
        linestyle="--",
        label="Uncertainty cone",
    )

    # --- second legend ---
    legend_track = ax1.legend(
        handles=[track_handle, cone_handle],
        loc="upper right",
        fontsize=7,
        frameon=True,
    )

    ax1.add_artist(legend_track)

    ax1.set_title(
        "Geographic exposure (wind buffers)",
        fontsize=9,
        fontweight="bold",
    )

    plot_bullseye_exposures(
        template_df,
        df_exp_plot,
        label_col="adm_label",
        id_col="adm1_src",
        speed_col="speed_kmh",
        min_font=8,
        max_font=20,
        speeds_order=colors.keys(),
        colors=colors,
        speed_unit="km/h",
        ax=ax2,
        fig_size=fig_size,
    )

    ax2.set_title(
        "Population exposure by region",
        fontsize=9,
        fontweight="bold",
    )

    fig.suptitle(
        f'Madagascar: exposure to "{fc_details["cyclone_name"]}" wind speed',
        fontsize=12,
        y=0.98,
    )

    fig.text(
        0.5,
        0.93,
        f"Forecast issued: {issued_time_str} (EAT) • {forecast_label}",
        ha="center",
        fontsize=10,
        style="italic",
    )

    fig.subplots_adjust(wspace=-0.1)

    return fig, (ax1, ax2)
```

```python
plot_both_exp(
    gdf_tracks_plot=records,
    gdf_buffers_plot=gdf_buffers,
    df_exp_plot=df_exp,
    forecast_label="Most likely track",
)
```

```python
plot_both_exp(
    gdf_tracks_plot=gdf_shift_tracks[
        gdf_shift_tracks["shift_deg"] == best_row["shift_deg"]
    ],
    gdf_buffers_plot=best_buffers,
    df_exp_plot=df_exp_adm1_best,
    forecast_label="Lowest-exposure track",
)
```

```python
plot_both_exp(
    gdf_tracks_plot=gdf_shift_tracks[
        gdf_shift_tracks["shift_deg"] == worst_row["shift_deg"]
    ],
    gdf_buffers_plot=worst_buffers,
    df_exp_plot=df_exp_adm1_worst,
    forecast_label="Highest-exposure track",
)
```

```python
sorted(speeds_kmh, reverse=True)
```

```python
df_exp_pivot = df_exp.merge(adm1[["adm1_src", "adm1_name"]]).pivot(
    index="adm1_name", columns="speed_kmh", values="pop_exposed"
)
```

```python
df_exp_pivot["≥ 118 km/h"] = df_exp_pivot[118]
df_exp_pivot["≥ 88 km/h"] = df_exp_pivot[88] - df_exp_pivot[118]
df_exp_pivot["≥ 62 km/h"] = df_exp_pivot[62] - df_exp_pivot[88]
df_exp_pivot["≥ 51 km/h"] = df_exp_pivot[51] - df_exp_pivot[62]

df_exp_pivot = df_exp_pivot.sort_values(51)
```

```python
df_exp_pivot
```

```python
colors_bar = {f"≥ {x} km/h": y for x, y in colors.items()}
```

```python
colors_bar
```

```python
fig, ax = plt.subplots(dpi=200, figsize=(10, 7))

cols = [f"≥ {x} km/h" for x in speeds_kmh]

df_exp_pivot[df_exp_pivot[51] > 0][cols].plot.bar(
    ax=ax, color=colors_bar, stacked=True, alpha=0.7
)

[ax.spines[x].set_visible(False) for x in ["top", "right"]]
ax.set_xlabel("Region")
ax.set_ylabel("Population exposed")
ax.set_title(
    f'Madagascar: exposure to "{fc_details["cyclone_name"]}" wind speed\n'
    f"Forecast issued: {issued_time_str} (EAT) • Most likely track"
)

ax.legend(title="Wind speed")

ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
```

```python
dt_utc = datetime.fromisoformat(fc_details["reference_time"])
```

```python
issued_time_str = dt_utc.strftime("%Y%m%dT%H%MZ")
```

```python
issued_time_str
```

```python
df_out = (
    df_exp.merge(adm1[["adm1_src", "adm1_name"]])
    .pivot(
        columns="speed_kmh",
        index=["adm1_src", "adm1_name"],
        values="pop_exposed",
    )
    .reset_index()
)
df_out.columns.name = None
df_out = df_out.sort_values("adm1_name")
df_out = df_out.rename(columns={x: f"exp_{x}_kmh" for x in [51, 62, 88, 118]})
df_out
```

```python
out_path = f'temp/{fc_details["cyclone_name"].lower()}_issued_{issued_time_str}_mdg_adm1_exposure.csv'
out_path
```

```python
df_out.to_csv(out_path, index=False)
```

```python

```
