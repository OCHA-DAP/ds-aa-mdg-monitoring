import pandas as pd

from src.constants import ISO3
from src.utils.db_utils import get_engine


def fetch_polygon_data(iso3: str = ISO3, adm_level: int = 1) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM public.polygon
    WHERE iso3 = '{iso3.upper()}'
    AND adm_level = {adm_level}
    """
    return pd.read_sql(query, get_engine(stage="prod"))
