import os
from typing import Literal

from dotenv import load_dotenv
from sqlalchemy import create_engine, insert

load_dotenv()

DSCI_AZ_DB_DEV_PW = os.getenv("DSCI_AZ_DB_DEV_PW")
DSCI_AZ_DB_PROD_PW = os.getenv("DSCI_AZ_DB_PROD_PW")

DSCI_AZ_DB_PROD_UID = os.getenv("DSCI_AZ_DB_PROD_UID")
DSCI_AZ_DB_DEV_UID = os.getenv("DSCI_AZ_DB_DEV_UID")

DSCI_AZ_DB_PROD_HOST = os.getenv("DSCI_AZ_DB_PROD_HOST")
DSCI_AZ_DB_DEV_HOST = os.getenv("DSCI_AZ_DB_DEV_HOST")

AZURE_DB_BASE_URL = "postgresql+psycopg2://{uid}:{pw}@{db_host}/postgres"


def get_engine(stage: Literal["dev", "prod"] = "dev"):
    """
    Create a SQLAlchemy engine for connecting to Azure SQL Database.

    Parameters
    ----------
    stage : Literal["dev", "prod"], optional
        Environment stage to connect to, by default "dev"

    Returns
    -------
    sqlalchemy.engine.Engine
        SQLAlchemy engine configured with the appropriate connection URL

    Raises
    ------
    ValueError
        If the provided stage is neither "dev" nor "prod"
    """
    if stage == "dev":
        url = AZURE_DB_BASE_URL.format(
            uid=DSCI_AZ_DB_DEV_UID,
            pw=DSCI_AZ_DB_DEV_PW,
            db_host=DSCI_AZ_DB_DEV_HOST,
        )
    elif stage == "prod":
        url = AZURE_DB_BASE_URL.format(
            uid=DSCI_AZ_DB_PROD_UID,
            pw=DSCI_AZ_DB_PROD_PW,
            db_host=DSCI_AZ_DB_PROD_HOST,
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")
    return create_engine(url)


def postgres_upsert(table, conn, keys, data_iter, constraint=None):
    """
    Perform an upsert (insert or update) operation on a PostgreSQL table. Adapted from:
    https://stackoverflow.com/questions/55187884/insert-into-postgresql-table-from-pandas-with-on-conflict-update # noqa: E501

    Parameters
    ----------
    table : sqlalchemy.sql.schema.Table
        The SQLAlchemy Table object where the data will be inserted or updated.
    conn : sqlalchemy.engine.Connection
        The SQLAlchemy connection object used to execute the upsert operation.
    keys : list of str
        The list of column names used as keys for the upsert operation.
    data_iter : iterable
        An iterable of tuples or lists containing the data to be inserted or
        updated.
    constraint_name : str
        Name of the uniqueness constraint

    Returns
    -------
    None
    """
    if not constraint:
        constraint = f"{table.table.name}_unique"
    data = [dict(zip(keys, row)) for row in data_iter]
    insert_statement = insert(table.table).values(data)
    upsert_statement = insert_statement.on_conflict_do_update(
        constraint=constraint,
        set_={c.key: c for c in insert_statement.excluded},
    )
    conn.execute(upsert_statement)
    return
