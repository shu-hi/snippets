from abc import ABC, abstractmethod
import pandas as pd
import duckdb
import func
import re

envs = func.get_envs()
class UnifiedData:

    def __init__(self):

        self.con = duckdb.connect()

        # postgres extension
        self.con.execute("INSTALL postgres")
        self.con.execute("LOAD postgres")

    # ------------------------
    # util
    # ------------------------

    def _safe_table(self, name):

        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError("invalid table name")

        return name

    # ------------------------
    # postgres
    # ------------------------

    def attach_postgres(self):
        db_user = os.getenv("DB_USER")
        db_pass = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        self.con.execute(f"""
        ATTACH '{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}' AS pg (TYPE postgres)
        """)

    # ------------------------
    # CSV
    # ------------------------

    def load_csv(self, table, path):

        table = self._safe_table(table)

        self.con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * FROM read_csv_auto('{path}')
        """)

    # ------------------------
    # Google Sheets (CSV export)
    # ------------------------

    def load_sheet(self, table, path):

        table = self._safe_table(table)

        self.con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * FROM read_csv_auto('{path}')
        """)

    # ------------------------
    # realtime sheet view
    # ------------------------

    def view_sheet(self, table, path):

        table = self._safe_table(table)
        self.con.execute(f"""
        CREATE OR REPLACE VIEW {table} AS
        SELECT * FROM read_csv_auto('{path}')
        """)

    # ------------------------
    # pandas table
    # ------------------------

    def register_df(self, table, df):

        table = self._safe_table(table)

        self.con.register(table, df)

    # ------------------------
    # query
    # ------------------------

    def query(self, sql):

        return self.con.execute(sql).fetchdf()