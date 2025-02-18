import io
import os
from typing import Literal

import pandas as pd
from azure.storage.blob import ContainerClient

PROJECT_PREFIX = "ds-aa-mdg-monitoring"
DEV_BLOB_SAS = os.getenv("DSCI_AZ_BLOB_DEV_SAS_WRITE")
PROD_BLOB_SAS = os.getenv("DSCI_AZ_BLOB_PROD_SAS")


def get_container_client(
    container_name: str = "projects", prod_dev: Literal["prod", "dev"] = "dev"
):
    sas = DEV_BLOB_SAS if prod_dev == "dev" else PROD_BLOB_SAS
    container_url = (
        f"https://imb0chd0{prod_dev}.blob.core.windows.net/"
        f"{container_name}?{sas}"
    )
    return ContainerClient.from_container_url(container_url)


def load_blob_data(
    blob_name,
    prod_dev: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
):
    container_client = get_container_client(
        prod_dev=prod_dev, container_name=container_name
    )
    blob_client = container_client.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    return data


def load_csv_from_blob(
    blob_name,
    prod_dev: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
    **kwargs,
):
    blob_data = load_blob_data(
        blob_name, prod_dev=prod_dev, container_name=container_name
    )
    return pd.read_csv(io.BytesIO(blob_data), **kwargs)
