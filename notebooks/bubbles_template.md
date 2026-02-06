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

# Population aggregation

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd
from tqdm.auto import tqdm

from src.monitoring.plotting import (
    wrap_text,
    build_circle_template,
    plot_template_circles,
)
from src.utils.blob_utils import PROJECT_PREFIX
```

```python
adm1 = stratus.codab.load_codab_from_fieldmaps(iso3="mdg", admin_level=1)
```

```python
adm1
```

```python
blob_name = "worldpop/pop_count/global_pop_2026_CN_1km_R2025A_UA_v1.tif"
da_wp = stratus.open_blob_cog(blob_name, container_name="raster")
```

```python
da_wp_clip = (
    da_wp.rio.clip(adm1.geometry, all_touched=True)
    .squeeze(drop=True)
    .compute()
)
da_wp_clip = da_wp_clip.where(da_wp_clip > 0)
```

```python
da_wp_clip
```

```python
da_wp_clip.plot()
```

```python
dicts = []
for pcode, row in tqdm(adm1.set_index("adm1_src").iterrows(), total=len(adm1)):
    try:
        da_clip = da_wp_clip.rio.clip([row.geometry])
    except NoDataInBounds as e:
        print(f"no pop found for {pcode}")
        continue
    dicts.append(
        {"pcode": pcode, "pop": int(da_clip.where(da_clip > 0).sum())}
    )
```

```python
df_pop = pd.DataFrame(dicts)
```

```python
df_pop
```

```python
df_pop["pop"].sum()
```

```python
gdf_admin = adm1.merge(df_pop.rename(columns={"pcode": "adm1_src"})).rename(
    columns={"pop": "pop_total"}
)
```

```python
gdf_admin["adm_label"] = gdf_admin["adm1_name"].apply(
    wrap_text, max_len=9, break_anywhere=True
)
```

```python
template_df = build_circle_template(
    gdf_admin,
    crs_equal_area="EPSG:3857",
    area_per_person=40000,  # adjust this until bubbles look right
    id_col="adm1_src",
)
```

```python
template_df = template_df.merge(gdf_admin[["adm1_src", "adm_label"]])
```

```python
plot_template_circles(
    template_df,
    label_col="adm_label",
    min_font=6,
    max_font=20,
)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/plotting/adm1_template.parquet"
stratus.upload_parquet_to_blob(template_df, blob_name)
```

```python

```
