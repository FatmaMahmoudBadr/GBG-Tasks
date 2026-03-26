import streamlit as st
from agent import build_agent

st.set_page_config(page_title="PostgreSQL AI Chat", layout="wide")

st.title("Chat with PostgreSQL Database")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask a database question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        container = st.empty()
        full_response = ""

        agent = build_agent(prompt)

        for step in agent.stream(
            {"messages": st.session_state.messages},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
            if "messages" in step:
                content = step["messages"][-1].content

                if content:
                    full_response = content
                    container.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )