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

_CSS = """
body{font-family:Helvetica,Arial,sans-serif;font-size:14px;color:#222;margin:0;padding:0;}
h1,h2{font-family:Arvo,"Helvetica Neue",Helvetica,Arial,sans-serif;font-weight:normal;}
h2{color:#1a6faf;border-bottom:2px solid #1a6faf;padding-bottom:6px;}
p{font-family:"Source Sans Pro","Helvetica Neue",Helvetica,Arial,sans-serif;}
img{max-width:100%;}
.disclaimer{font-size:11px;padding:6px 10px;background-color:#F0F0F0;
            color:#000;font-style:italic;}
.disclaimer p{margin:0;}
hr{border:none;border-top:1px solid #ccc;margin:16px 0;}
.note{font-size:12px;color:#777;margin-top:4px;}
"""


def _build_body(df, fig) -> tuple[str, str]:
    middle_date = str(df["valid_date"].unique()[1])
    df_grouped = df.groupby("pcode")["mean"].sum().reset_index()
    obsv_trigger = (
        "ACTIVÉ"
        if df_grouped["mean"].max() > RAIN_THRESH
        else "PAS ACTIVÉ"
    )

    plot_data = BytesIO()
    fig.savefig(plot_data, format="png", bbox_inches="tight")
    plot_data.seek(0)
    plot_b64 = base64.b64encode(plot_data.read()).decode()

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_CSS}</style></head>
<body>

<h2>Précipitations autour de {middle_date}</h2>

<div class="disclaimer">
  <p>
    Cet e-mail est purement consultatif et ne sert pas d'avis officiel
    pour le cadre d'action anticipatoire.
  </p>
</div>

<p>Chers collègues,</p>
<p>
  De nouvelles données observationnelles satellitaires des précipitations
  (IMERG) viennent d'être émises par NASA.
</p>
<p>
  Déclencheur observationnel : <strong>{obsv_trigger}</strong>
</p>
<p>Un graphique des observations est présenté ci-dessous.</p>
<img src="data:image/png;base64,{plot_b64}" width="100%"/>
<p>
  Le code utilisé pour produire cette alerte est disponible sur GitHub
  <a href="https://github.com/OCHA-DAP/ds-aa-mdg-monitoring">ici</a>.
</p>

<hr>

<h2>Détails</h2>
<p>
  Pour plus de détails, veuillez consulter
  <a href="https://reliefweb.int/report/madagascar/cadre-de-laction-anticipatoire-pilote-madagascar-cyclones-version-finale-du-13-decembre-2024">le Cadre de l'action anticipatoire - Pilote à Madagascar pour les Cyclones</a>.
</p>


</body>
</html>"""

    return html, middle_date


def send_info_email(df, fig, test: bool = False) -> int:
    """Send the rainfall monitoring email. Returns the Listmonk campaign ID."""
    client = ListmonkClient.from_env()
    html_body, middle_date = _build_body(df, fig)
    prefix = "[test] " if test else ""
    subject = (
        f"{prefix}Action anticipatoire Madagascar"
        f" – précipitations autour de {middle_date}"
    )
    eat = timezone(timedelta(hours=3))
    ts = datetime.now(eat).strftime("%Y%m%dT%H%M")
    campaign_name = (
        f"{'[test]-' if test else ''}"
        f"mdg-cyclone-rainfall-{middle_date}-{ts}"
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
