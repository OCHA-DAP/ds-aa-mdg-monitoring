# Monitoring for Madagascar cyclones anticipatory action framework: Rainfall using IMERG v7.
Adding pipeline to plot IMERG for most recent three consecutive days, and send email.

## Mailing lists
Mailing lists are in blob at ds-aa-mdg-monitoring/monitoring. To toggle between the test list test_distribution_list.csv and actual list distribution_list.csv, use the env var TEST_LIST, which is set as a variable in this repository. The only way it will send to the actual mailing list is if TEST_LIST is set to False. Otherwise (including if it isn't set), the test mailing will be used, to avoid accidentally mailing the full list when debugging.

The lists have columns:

email: to be filled with peoples' plain emails
name: the display name that will be used (although Outlook will fill this in automatically for emails in your organization)
info: set to to or cc to determine how they are listed in the email. If set to anything other than to or cc, they will not be sent the email
More columns can be added (e.g. trigger) if we want to add other email notification types that go out at different times to different lists.

## Schedule
Sent every day at 4pm UTC, after the IMERG raster stats land (Databricks "Run IMERG", 14:40 UTC). There is no backfilling of missed past dates.

Runs as the Databricks job **MDG IMERG Monitoring**, defined in `databricks.yml`
(one task running `pipelines/monitor_imerg.py` unchanged through
`databricks/run_task.py`). The `dev` target sends to the Listmonk test list;
the `prod` target (`test_email=false`) to the real list.

```shell
databricks bundle validate -t prod -p DEFAULT
databricks bundle deploy   -t prod -p DEFAULT        # (re)deploy the live job
databricks bundle run mdg_imerg_monitoring -t prod -p DEFAULT --params test_email=true
```

The GitHub Actions workflow remains as a manual fallback (`workflow_dispatch`); its cron is gone.
The script also honours `MONITORING_DATE` and `TEST_EMAIL` env vars (equivalent to `--date` / `--test`).

## Running the script
Run locally with python pipelines/monitor_imerg.py.

Takes optional argument command line argument date, which is the center date of the three consecutive days. This defaults to two days before today, for which we should have all raster stats. This is also included as an argument in the GH Action workflow_dispatch.

To run on GH actions, use the main branch.

## Trigger
The threshold is >= 300 mm in any region (averaged over region, summed over three days). This is set as RAIN_THRESH in src/constants.py. 
