import base64
import os
import re
import smtplib
import ssl
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import make_msgid
from io import BytesIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from html2text import html2text
from jinja2 import Environment, FileSystemLoader

from src.constants import RAIN_THRESH
from src.utils import blob_utils

load_dotenv()

EMAIL_HOST = os.getenv("DS_AWS_EMAIL_HOST")
EMAIL_PORT = int(os.getenv("DS_AWS_EMAIL_PORT", 465))
EMAIL_PASSWORD = os.getenv("DS_AWS_EMAIL_PASSWORD")
EMAIL_USERNAME = os.getenv("DS_AWS_EMAIL_USERNAME")
EMAIL_ADDRESS = os.getenv("DS_AWS_EMAIL_ADDRESS")

TEST_LIST = os.getenv("TEST_LIST")
if TEST_LIST == "False":
    TEST_LIST = False
else:
    TEST_LIST = True

TEMPLATES_DIR = (
    Path(__file__).resolve().parents[2] / "email_assets" / "templates"
)
STATIC_DIR = Path(__file__).resolve().parents[2] / "email_assets" / "static"


def open_static_image(filename: str) -> str:
    filepath = STATIC_DIR / filename
    with open(filepath, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image


def get_distribution_list() -> pd.DataFrame:
    """Load distribution list from blob storage."""
    if TEST_LIST:
        blob_name = f"{blob_utils.PROJECT_PREFIX}/monitoring/test_distribution_list.csv"  # noqa
    else:
        blob_name = (
            f"{blob_utils.PROJECT_PREFIX}/monitoring/distribution_list.csv"
        )
    return blob_utils.load_csv_from_blob(blob_name)


def is_valid_email(email):
    # Define a regex pattern for validating an monitoring
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    # Use the re.match() method to check if the monitoring matches the pattern
    if re.match(email_regex, email):
        return True
    else:
        return False


def send_info_email(df: pd.DataFrame, fig):
    middle_date = str(df["valid_date"].unique()[1])
    df_grouped = df.groupby("pcode")["mean"].sum().reset_index()
    obsv_trigger = (
        "ACTIVÉ" if df_grouped["mean"].max() > RAIN_THRESH else "PAS ACTIVÉ"
    )

    distribution_list = get_distribution_list()
    valid_distribution_list = distribution_list[
        distribution_list["email"].apply(is_valid_email)
    ]
    invalid_distribution_list = distribution_list[
        ~distribution_list["email"].apply(is_valid_email)
    ]
    if not invalid_distribution_list.empty:
        print(
            f"Invalid emails found in distribution list: "
            f"{invalid_distribution_list['info'].tolist()}"
        )
    to_list = valid_distribution_list[valid_distribution_list["info"] == "to"]
    cc_list = valid_distribution_list[valid_distribution_list["info"] == "cc"]

    environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    template_name = "informational"
    template = environment.get_template(f"{template_name}.html")
    msg = EmailMessage()
    msg.set_charset("utf-8")
    msg["Subject"] = (
        f"Action anticipatoire Madagascar – précipitations autour de {middle_date}"  # noqa
    )
    msg["From"] = Address(
        "Centre de données humanitaires OCHA",
        EMAIL_ADDRESS.split("@")[0],
        EMAIL_ADDRESS.split("@")[1],
    )
    msg["To"] = [
        Address(
            row["name"],
            row["email"].split("@")[0],
            row["email"].split("@")[1],
        )
        for _, row in to_list.iterrows()
    ]
    msg["Cc"] = [
        Address(
            row["name"],
            row["email"].split("@")[0],
            row["email"].split("@")[1],
        )
        for _, row in cc_list.iterrows()
    ]

    plot_cid = make_msgid(domain="humdata.org")
    chd_banner_cid = make_msgid(domain="humdata.org")
    ocha_logo_cid = make_msgid(domain="humdata.org")

    html_str = template.render(
        obsv_trigger=obsv_trigger,
        middle_date=middle_date,
        plot_cid=plot_cid[1:-1],
        chd_banner_cid=chd_banner_cid[1:-1],
        ocha_logo_cid=ocha_logo_cid[1:-1],
    )
    body_text_str = html2text(html_str)
    # include preview text at the beginning for certain email clients
    preview_text_str = f"""
    Action anticipatoire Madagascar - précipitations autour de {middle_date}\n\n
    Déclencheur observationnel : {obsv_trigger}\n
    """  # noqa
    text_str = preview_text_str + body_text_str
    msg.set_content(text_str)
    msg.add_alternative(html_str, subtype="html")

    # add in plot
    image_data = BytesIO()
    fig.savefig(image_data, format="png", bbox_inches="tight")
    image_data.seek(0)
    msg.get_payload()[1].add_related(
        image_data.read(), "image", "png", cid=plot_cid
    )

    # add in other images
    for filename, cid in zip(
        ["centre_banner.png", "ocha_logo_wide.png"],
        [chd_banner_cid, ocha_logo_cid],
    ):
        img_path = STATIC_DIR / filename
        with open(img_path, "rb") as img:
            msg.get_payload()[1].add_related(
                img.read(), "image", "png", cid=cid
            )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.sendmail(
            EMAIL_ADDRESS,
            to_list["email"].tolist() + cc_list["email"].tolist(),
            msg.as_string(),
        )
