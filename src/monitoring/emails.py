"""Build and send Madagascar rainfall monitoring emails via Listmonk."""

import base64
from datetime import datetime, timedelta, timezone
from io import BytesIO

from ocha_relay.listmonk import ListmonkClient

from src.constants import (
    LISTMONK_LIST_ID,
    LISTMONK_LIST_ID_TEST,
    RAIN_THRESH,
)

# Inline styles copied from Listmonk template 8's own Contact section and
# notice box, so the body matches the header and footer around it. Inline
# because mail clients drop <style> blocks inside the body.
_FONT = "font-family:Helvetica,Arial,sans-serif;"
_H2 = _FONT + "font-size:24px;margin:0 0 12px 0;font-weight:600;color:#222222;"
_P = _FONT + "font-size:16px;margin:0 0 12px 0;color:#222222;"
_NOTICE = (
    _FONT + "font-size:14px;padding:6px 10px;background-color:#F0F0F0;"
    "color:#000000;font-style:italic;margin-bottom:12px;"
)
_HR = "border:none;border-top:1px solid #dddddd;margin:24px 0;"
_LINK = "color:#0066cc;text-decoration:underline;"

_MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
    "août", "septembre", "octobre", "novembre", "décembre",
]


def _date_fr(date) -> str:
    """2026-10-19 -> 19 octobre 2026"""
    d = datetime.strptime(str(date)[:10], "%Y-%m-%d")
    return f"{d.day} {_MONTHS_FR[d.month - 1]} {d.year}"


def _build_body(df, fig) -> tuple[str, str]:
    middle_date = str(df["valid_date"].unique()[1])
    df_grouped = df.groupby("pcode")["mean"].sum().reset_index()
    reached = df_grouped["mean"].max() > RAIN_THRESH
    obsv_trigger = "ATTEINT" if reached else "NON ATTEINT"
    trigger_color = "#c0392b" if reached else "#0066cc"

    plot_data = BytesIO()
    fig.savefig(plot_data, format="png", bbox_inches="tight")
    plot_data.seek(0)
    plot_b64 = base64.b64encode(plot_data.read()).decode()

    html = f"""<div style="{_FONT}color:#222;">

<h2 style="{_H2}">Précipitations autour du {_date_fr(middle_date)}</h2>

<div style="{_NOTICE}">
  <p style="margin:0;">
    Cet e-mail est purement consultatif et ne sert pas d'avis officiel
    pour le cadre d'action anticipatoire.
  </p>
</div>

<p style="{_P}">Chers collègues,</p>
<p style="{_P}">
  De nouvelles données observationnelles satellitaires des précipitations
  (IMERG) viennent d'être émises par NASA.
</p>
<p style="{_P}">
  Déclencheur observationnel :
  <strong style="color:{trigger_color};">{obsv_trigger}</strong>
</p>
<p style="{_P}">Un graphique des observations est présenté ci-dessous.</p>
<img src="data:image/png;base64,{plot_b64}" width="100%" style="max-width:100%;"/>
<p style="{_P}">
  Le code utilisé pour produire cette alerte est disponible sur GitHub
  <a href="https://github.com/OCHA-DAP/ds-aa-mdg-monitoring" style="{_LINK}">ici</a>.
</p>

<hr style="{_HR}">

<h2 style="{_H2}">Détails</h2>
<p style="{_P}">
  Pour plus de détails, veuillez consulter
  <a href="https://reliefweb.int/report/madagascar/cadre-de-laction-anticipatoire-pilote-madagascar-cyclones-version-finale-du-13-decembre-2024" style="{_LINK}">le Cadre de l'action anticipatoire - Pilote à Madagascar pour les Cyclones</a>.
</p>

</div>"""

    return html, middle_date


def send_info_email(df, fig, test: bool = False) -> int:
    """Send the rainfall monitoring email. Returns the Listmonk campaign ID."""
    client = ListmonkClient.from_env()
    html_body, middle_date = _build_body(df, fig)
    prefix = "[test] " if test else ""
    subject = (
        f"{prefix}Action anticipatoire Madagascar"
        f" – précipitations autour du {_date_fr(middle_date)}"
    )
    eat = timezone(timedelta(hours=3))
    ts = datetime.now(eat).strftime("%Y%m%dT%H%M")
    # "[fr]" in the name switches Listmonk template 8's footer to French
    campaign_name = (
        f"{'[test]-' if test else ''}"
        f"[fr]-mdg-cyclone-rainfall-{middle_date}-{ts}"
    )
    list_id = LISTMONK_LIST_ID_TEST if test else LISTMONK_LIST_ID
    campaign_id = client.create_campaign(
        name=campaign_name,
        subject=subject,
        body=html_body,
        list_ids=[list_id],
    )
    client.send_campaign(campaign_id, skip_confirmation=True)
    return campaign_id
