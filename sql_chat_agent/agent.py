from config import OPENROUTER_API_KEY
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from database import get_database
from fewshots import few_shots

def get_llm():

    return init_chat_model(
        model="openai/gpt-4o-mini",
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0,
        max_tokens=1024,
    )


def build_agent(question: str):

    db = get_database()
    llm = get_llm()

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

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
    return create_agent(llm, tools, system_prompt=system_prompt)
