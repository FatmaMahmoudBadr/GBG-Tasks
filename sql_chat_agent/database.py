from config import DB_URL
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
import streamlit as st


@st.cache_resource
def get_database():
    engine = create_engine(
        DB_URL,
        connect_args={"options": "-c statement_timeout=5000"},
        pool_pre_ping=True
    )

    return SQLDatabase(engine)