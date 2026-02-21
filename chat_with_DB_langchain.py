import os
from typing import Literal
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")

st.set_page_config(page_title="DB Chat", layout="wide")
st.title("Chat with Database ")

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

engine = get_engine()

@st.cache_resource
def get_schema():

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
            current_table=""
            for row in result:
                table_name, column_name = row[0], row[1]
                if table_name!= current_table:
                    schema_str += f".\nTable: {table_name}\nColumns: {column_name}"
                    current_table = table_name
                    continue
                schema_str += f", {column_name}"

    except Exception as e:
        st.error(f"ERROR reading schema: {e}")
    # print(schema_str)
    return schema_str

def run_query(query):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        return str(e)

def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0
    )

llm = get_llm()

sql_template = """You are an expert PostgreSQL SQL query generator.

Your task is to generate a valid PostgreSQL SELECT query that answers the user's question based ONLY on the provided table schema.

OUTPUT REQUIREMENTS
- Return ONLY the raw SQL query.
- The query MUST be written in a single line.
- Do NOT include explanations.
- Do NOT include comments.
- Do NOT format in markdown.
- Do NOT wrap in ```sql.

DATABASE DIALECT
- Database system: PostgreSQL.
- Use PostgreSQL syntax ONLY.
- Do NOT use SQLite functions such as STRFTIME.
- When extracting date parts, use: EXTRACT(YEAR FROM column).
- When using EXTRACT, ALWAYS cast the column using ::timestamp.

TABLE SCHEMA
{schema}

USER QUESTION
{question}

CRITICAL SQL FORMATTING RULES
- Always wrap table names in double quotes.
- Always wrap column names in double quotes.
- Do NOT use table aliases.
- If selecting columns with identical names from different tables, ALWAYS use column aliases.
- Column aliases MUST be wrapped in double quotes.
- Do NOT use column aliases inside GROUP BY.
- Repeat the full expression inside GROUP BY instead of using the alias.

STRICT QUERY RULES
- Only generate SELECT queries.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.
- Use ONLY the tables and columns provided in the schema.
- Do NOT hallucinate tables or columns.
- Always use proper JOIN conditions based on foreign keys.
- If aggregation is required, use GROUP BY correctly.
- If the query may return many rows, add LIMIT 20 at the end.
- Do NOT add LIMIT 20 if the query already contains LIMIT 1.
- If a required column does not exist in the schema, return:
  ERROR: Column not found in schema.
"""


answer_template = """
You are a professional data analyst assistant.

USER QUESTION:
{question}

SQL QUERY EXECUTED:
{query}

SQL RESULT (first rows only):
{data}

INSTRUCTIONS:
1. Answer the user's question ONLY based on the SQL result.
2. Do NOT hallucinate additional data.
3. If the result is empty, say:
   "No matching records were found."
4. If only 20 rows are shown, inform the user that results may be limited.
5. If aggregated data is present, clearly explain what it means.
6. Format currency values clearly (e.g., $123.45).
7. Be concise, clear, and professional.
8. Do NOT mention SQL errors unless explicitly provided.

Provide a natural language answer.
"""


sql_prompt = ChatPromptTemplate.from_template(sql_template)
answer_prompt = ChatPromptTemplate.from_template(answer_template)
schema = get_schema()
sql_chain = sql_prompt | llm|StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()


# ---------------- USER INPUT ----------------
user_question = st.text_input("Ask a question about the database:")

if user_question:

    # -------- Generate SQL --------
    query = sql_chain.invoke(
        {"question": user_question,
         "schema": schema},
         config={"run_name": "SQL_Query_Generation"}
    ).replace("```sql", "").replace("```", "").strip()

    st.subheader("Generated SQL")
    st.code(query, language="sql")

    # -------- Execute SQL --------
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate"]
    if any(word in query.lower() for word in forbidden):
        st.error("Unsafe query detected.")
        st.stop()
    df = run_query(query)

    if isinstance(df, str):
        st.error(f"SQL Execution Error:\n{df}")
    else:
        st.subheader("SQL Result")
        st.dataframe(df)

        # -------- Generate Answer --------
        answer = answer_chain.invoke(
            {
                "question": user_question,
                "data": df,
                "query": query
            },
            config={"run_name": "Answer_Generation"}
        )

        st.subheader("Answer")
        st.write(answer)
