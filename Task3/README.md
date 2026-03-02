# Chat with Database (LLM + PostgreSQL)

A Streamlit application that allows users to query a PostgreSQL database using natural language.  
The system uses Google Gemini (via LangChain) to generate SQL queries and explain results in natural language.

---

## Features

- Natural language → SQL generation
- PostgreSQL schema awareness
- Safe query enforcement (SELECT only)
- Automatic schema extraction
- Clean architecture (modular design)
- LLM-generated professional answers
- Streamlit UI

---

## Architecture

User Question
      ↓
LLM generates SQL (schema-aware)
      ↓
SQL validation (safety check)
      ↓
Database execution
      ↓
LLM explains result

---

## Project Structure

app/
- config.py → Environment configuration
- database.py → Database connection, execution & Schema extraction
- llm.py → LLM initialization & Prompt templates
- services.py → Utility functions

streamlit_app.py → LangChain pipelines & Main Streamlit application

---

## Setup

### 1️⃣ Clone the repository
git clone <repo_url>
cd Task3

### 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Configure environment variables
Create .env file:

GOOGLE_API_KEY=your_google_api_key
DB_URL=postgresql://username:password@localhost:5432/your_db

### ▶️ Run the Application
streamlit run streamlit_app.py
