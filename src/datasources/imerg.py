from typing import List

import pandas as pd

from src.utils.db_utils import get_engine


def fetch_imerg_data(
    pcodes: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Fetch IMERG data for a list of PCODES and a date range

    Parameters
    ----------
    pcodes
        List of PCODES
    start_date
        Start date
    end_date
        End date

    Returns
    -------
    pd.DataFrame
        DataFrame with IMERG data
    """
    pcodes_query_str = ", ".join([f"'{p}'" for p in pcodes])
    query = f"""
    SELECT *
    FROM public.imerg
    WHERE
        valid_date BETWEEN '{start_date.date()}' AND '{end_date.date()}'
        AND pcode IN ({pcodes_query_str})
    """
    return pd.read_sql(query, get_engine(stage="prod"))
