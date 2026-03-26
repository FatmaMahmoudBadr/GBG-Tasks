# Chat with Database Agent(PostgreSQL + SQLDatabaseToolkit + few-shot retrieval)

An intelligent SQL agent that converts natural language questions into **PostgreSQL queries and returns clear, human-readable answers**.

Built with **LangChain, OpenRouter (LLMs), FAISS, and Streamlit**, this agent intelligently converts user questions into SQL queries, executes them, and returns human-readable answers.

---

## Features

- Natural Language to SQL using LLMs (GPT-4o-mini via OpenRouter)
- PostgreSQL database integration
- Few-shot retrieval using FAISS + Sentence Transformers
- Real-time streaming responses
- Safe query generation (SELECT-only queries)
- Interactive chat UI with Streamlit
- Context-aware examples to improve accuracy

---

## Project Structure

sql_chat_agent/
  - agent.py  →  **Builds the SQL agent and LLM configuration**
  - database.py  →  **PostgreSQL connection setup**
  - fewshots.py  →   **Fewshots retrieval using FAISS**
  - config.py    → **Environment variables loader**
  - streamlit_app.py   → **Streamlit chat interface**
  data/
    - fewshots.json   → **Example questions and SQL queries**
  - README.md

---

## Installation

### 1. Clone the repository

git clone https://github.com/FatmaMahmoudBadr/GBG-Tasks.git

cd GBG-Tasks/sql_chat_agent


### 2. Create a virtual environment

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate


### 3. Install dependencies

pip install -r requirements.txt


### 4. Environment Variables

Create a .env file in the root directory:

- DB_URL=your_postgresql_connection_string
- OPENROUTER_API_KEY=your_openrouter_api_key
---
### Running the App

streamlit run streamlit_app.py
