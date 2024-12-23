import pandas as pd

from src.datasources.imerg import fetch_imerg_data
from src.datasources.polygon import fetch_polygon_data

if __name__ == "__main__":
    print("Monitoring IMERG...")
    print("Getting polygon data...")
    adm_df = fetch_polygon_data()
    # set dates to three days before today
    dates = pd.date_range(
        end=pd.Timestamp.today() - pd.DateOffset(days=1), periods=3
    )
    print("Fetching IMERG data for the following dates:")
    print(f"{dates[0].date()} to {dates[-1].date()}")
    imerg_df = fetch_imerg_data(adm_df["pcode"].to_list(), dates[0], dates[-1])
    # check if the most recent date is in the dataframe
    assert (
        imerg_df["valid_date"].max() == dates[-1].date()
    ), "Most recent date not found"
    imerg_grouped = imerg_df.groupby("pcode")["mean"].sum()
    print(imerg_grouped)
