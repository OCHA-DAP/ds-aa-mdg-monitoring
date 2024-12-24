import argparse

import pandas as pd

from src.datasources.imerg import fetch_imerg_data
from src.datasources.polygon import fetch_polygon_data
from src.monitoring.emails import send_info_email
from src.monitoring.plotting import plot_rainfall


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IMERG monitoring for a specific date."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=(pd.Timestamp.today() - pd.DateOffset(days=2)).strftime(
            "%Y-%m-%d"
        ),
        help="Date to run for, in YYYY-MM-DD format (default is today). "
        "Script uses rainfall data for the three days before this date.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_date = pd.to_datetime(args.date)
    dates = pd.date_range(end=run_date + pd.DateOffset(days=1), periods=3)
    print(
        f"Monitoring IMERG for three days centered on {run_date.date()}: "
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

    imerg_df = imerg_df.merge(adm_df[["pcode", "name"]]).sort_values(
        ["valid_date", "name"]
    )

    print("Plotting...")
    fig, ax = plot_rainfall(imerg_df)

    print("Sending email...")
    send_info_email(imerg_df, fig)
