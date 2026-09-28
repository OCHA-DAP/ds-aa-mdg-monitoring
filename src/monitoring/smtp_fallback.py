"""SMTP fallback for when the shared Listmonk instance is unavailable.

Listmonk's app can be up (``/health`` responds) while every authenticated
endpoint (``campaigns``, ``subscribers``) hangs or times out. When that
happens, ``send_info_email`` falls back to plain SMTP via the same
@humdata.org identity used before the Listmonk migration, with recipients
loaded from the blob distribution lists (``test_distribution_list.csv`` /
``distribution_list.csv``) rather than Listmonk's subscriber lists, since
those can't be queried while Listmonk is down.
"""

import os
import smtplib
import ssl
from email.headerregistry import Address
from email.message import EmailMessage

from src.utils import blob_utils

EMAIL_HOST = os.getenv("DSCI_AWS_EMAIL_HOST")
EMAIL_PORT = int(os.getenv("DSCI_AWS_EMAIL_PORT", 465))
EMAIL_USERNAME = os.getenv("DSCI_AWS_EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("DSCI_AWS_EMAIL_PASSWORD")
EMAIL_ADDRESS = os.getenv("DSCI_AWS_EMAIL_ADDRESS")


def _get_distribution_list(test: bool):
    filename = "test_distribution_list.csv" if test else "distribution_list.csv"
    blob_name = f"{blob_utils.PROJECT_PREFIX}/monitoring/{filename}"
    return blob_utils.load_csv_from_blob(blob_name)


def send_via_smtp(subject: str, html_body: str, test: bool = False) -> None:
    """Send subject/html_body as a plain email, bypassing Listmonk."""
    distribution_list = _get_distribution_list(test)
    to_list = distribution_list[distribution_list["info"] == "to"]
    cc_list = distribution_list[distribution_list["info"] == "cc"]

    msg = EmailMessage()
    msg.set_charset("utf-8")
    msg["Subject"] = subject
    msg["From"] = Address(
        "Centre de données humanitaires OCHA",
        *EMAIL_ADDRESS.split("@"),
    )
    msg["To"] = [
        Address(row["name"], *row["email"].split("@"))
        for _, row in to_list.iterrows()
    ]
    if not cc_list.empty:
        msg["Cc"] = [
            Address(row["name"], *row["email"].split("@"))
            for _, row in cc_list.iterrows()
        ]
    msg.add_alternative(html_body, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.sendmail(
            EMAIL_ADDRESS,
            to_list["email"].tolist() + cc_list["email"].tolist(),
            msg.as_string(),
        )
