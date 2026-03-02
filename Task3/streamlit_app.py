import streamlit as st
from app.database import run_query, get_schema
from app.helper_functions import clean_sql, is_safe_query
from langchain_core.output_parsers import StrOutputParser
from app.llm import get_llm, sql_prompt, answer_prompt

llm = get_llm()

sql_chain = sql_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()

st.set_page_config(page_title="DB Chat", layout="wide")
st.title("Chat with Database")

schema = get_schema()

user_question = st.text_input("Ask a question about the database:")

if user_question:

    # Generate SQL
    query = sql_chain.invoke(
        {"question": user_question, "schema": schema},
        config={"run_name": "SQL_Query_Generation"}
    )

    query = clean_sql(query)

    st.subheader("Generated SQL")
    st.code(query, language="sql")

    if not is_safe_query(query):
        st.error("Unsafe query detected.")
        st.stop()

    # Execute query
    df = run_query(query)

    if isinstance(df, str):
        st.error(f"SQL Execution Error:\n{df}")
    else:
        st.subheader("SQL Result")
        st.dataframe(df)

        # Generate answer
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