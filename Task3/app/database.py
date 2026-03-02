import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
from .config import DB_URL


@st.cache_resource
def get_engine():
    return create_engine(DB_URL)


def run_query(query: str):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        return str(e)
    

@st.cache_resource
def get_schema():
    engine = get_engine()

    inspector_query = text("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)

    schema_str = ""

    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            current_table = ""

            for row in result:
                table_name, column_name = row[0], row[1]

                if table_name != current_table:
                    schema_str += f".\nTable: {table_name}\nColumns: {column_name}"
                    current_table = table_name
                    continue

                schema_str += f", {column_name}"

    except Exception as e:
        st.error(f"ERROR reading schema: {e}")

    return schema_str