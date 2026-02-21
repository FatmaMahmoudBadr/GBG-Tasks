import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from few_shots_rag import few_shots

# Load Environment
load_dotenv()

# Streamlit Config
st.set_page_config(page_title="PostgreSQL AI Chat", layout="wide")
st.title("Chat with PostgreSQL Database")

# Initialize LLM
model = init_chat_model(
    model="openai/gpt-4o-mini",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_tokens=1024,
)

DB_URL = os.getenv("DB_URL")

@st.cache_resource
def get_database():
    engine = create_engine(
        DB_URL,
        connect_args={"options": "-c statement_timeout=5000"},
        pool_pre_ping=True
    )
    return SQLDatabase(engine)

db = get_database()

# Create Agent
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

def build_agent(question: str):
    examples = few_shots(question)

    system_prompt = f"""
    You are an expert PostgreSQL SQL agent.

    Your job is to answer user questions by interacting with the SQL database using tools.

    DATABASE RULES:
    - Database system: PostgreSQL.
    - Use PostgreSQL syntax only.
    - When extracting date parts, use: EXTRACT(YEAR FROM column::timestamp).
    - Do NOT use SQLite-specific functions.

    QUERY RULES:
    - Only generate SELECT queries.
    - Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.
    - Always wrap table names in double quotes.
    - Always wrap column names in double quotes.
    - Do NOT use table aliases.
    - If selecting columns with identical names from different tables, use explicit column aliases.
    - Do NOT use column aliases inside GROUP BY.
    - Repeat the full expression inside GROUP BY.
    - If query may return many rows, limit to 10 rows.
    - Do NOT hallucinate tables or columns.
    - Use only tables and columns that exist in the schema.

    FEW-SHOT EXAMPLES:
    {examples}

    AGENT WORKFLOW:
    1. First, list available tables.
    2. Then inspect the schema of relevant tables.
    3. Generate a correct SQL query.
    4. Double-check the query before executing.
    5. If execution fails, rewrite and try again.
    6. After execution, return a clear final answer to the user.
    
    IMPORTANT
    - If the user greets you Do NOT use database tools and Respond politely and briefly.
    - If the question is not related to the database, respond: "I can only answer questions related to this database."
    - Never attempt to answer anything outside the database.
    - Never output raw SQL as the final answer.
    - Always return a natural-language answer based on the query results.
    """

    return create_agent(model, tools, system_prompt=system_prompt)

# Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a database question..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        agent = build_agent(prompt)

        for step in agent.stream(
            {"messages": st.session_state.messages},
            stream_mode="values",
        ):
            if "messages" in step:
                content = step["messages"][-1].content
                if content:
                    full_response = content
                    response_container.markdown(full_response)

        # Save assistant reply
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )