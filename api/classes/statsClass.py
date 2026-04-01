from abc import ABC, abstractmethod
import pandas as pd
import duckdb
import func
import re
import os
import json
envs = func.get_envs()
class UnifiedData:

    def __init__(self):
        self.con = duckdb.connect()
        # postgres extension
        self.con.execute("INSTALL postgres")
        self.con.execute("LOAD postgres")
        self.con.execute("SET force_download=true")
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
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")
        self.con.execute(f"""
            ATTACH 'dbname={db_name} user={db_user} password={db_pass} host={db_host} port={db_port}' 
            AS pg (TYPE postgres)
            """)

    # ------------------------
    # CSV
    # ------------------------
    def load_csv(self, path):
        table = self._safe_table('csv')
        self.con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * FROM read_csv_auto('{path}')
        """)
    # ------------------------
    # Google Sheets (CSV export)
    # ------------------------
    def load_sheet(self, path):
        spl=path.split('/')
        ext=sorted(spl,key=lambda word: len(word))[-1]if spl else None
        table = self._safe_table('gss')
        self.con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * FROM read_csv_auto('https://docs.google.com/spreadsheets/export?format=csv&id={ext}')
        """)

    # ------------------------
    # realtime sheet view
    # ------------------------

    def view_sheet(self, path):
        table = self._safe_table('gssv')
        self.con.execute(f"""
        CREATE OR REPLACE VIEW {table} AS
        SELECT * FROM read_csv_auto('{path}')
        """)

    # ------------------------
    # pandas table
    # ------------------------

    def register_df(self, df):
        table = self._safe_table('df')
        self.con.register(table, df)

    # ------------------------
    # query
    # ------------------------
    def query(self, sql):
        df = self.con.execute(sql).fetchdf()
        json_str = df.to_json(orient="records", date_format="iso")
        data = json.loads(json_str)
        return data