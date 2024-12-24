import argparse

import pandas as pd
from matplotlib import pyplot as plt

from src.datasources.imerg import fetch_imerg_data
from src.datasources.polygon import fetch_polygon_data
from src.email.plotting import plot_rainfall


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IMERG monitoring for a specific date."
    )
    parser.add_argument(
        "--today",
        type=str,
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="Date to run for, in YYYY-MM-DD format (default is today). "
        "Script uses rainfall data for the three days before this date.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_date = pd.to_datetime(args.today)
    dates = pd.date_range(end=run_date - pd.DateOffset(days=1), periods=3)
    print(
        f"Running IMERG for three days before {run_date.date()}: "
        f"{[str(x.date()) for x in dates]}"
    )

    print("Getting polygon data...")
    adm_df = fetch_polygon_data()

    print("Getting IMERG data...")
    imerg_df = fetch_imerg_data(adm_df["pcode"].to_list(), dates[0], dates[-1])

    # Check that all dates are found in the data
    missing_dates = [
        str(x.date())
        for x in dates
        if x.date() not in imerg_df["valid_date"].values
    ]

    if missing_dates:
        raise ValueError(
            "The following dates are missing from the IMERG data: "
            f"{missing_dates}"
        )

    # Merge and sort the data
    imerg_df = imerg_df.merge(adm_df[["pcode", "name"]]).sort_values(
        ["valid_date", "name"]
    )

    # Plot the rainfall data
    fig, ax = plot_rainfall(imerg_df)
    plt.show()
