# Bilingual Sentiment Analysis (Arabic & English)

This project implements a bilingual sentiment analysis system using LangChain and Google Gemini (2.5 Flash).
It supports both Arabic and English texts and classifies them into:
positive
negative
neutral
mixed

# Project Files

### sentiment_analysis_reasoning.py
Advanced reasoning-based sentiment analysis with structured explanations (English & Arabic).

### sentiment_analysis_reliable.py
Deterministic sentiment classifier (temperature = 0) for more stable and production-ready results.

### sentiment_reasoning_results.jsonl
Output results from the reasoning-based model.

### sentiment_reliable_results.jsonl
Output results from the reliable-based model.

# Technologies Used 
Python, LangChain, Google Gemini 2.5 Flash, Pydantic, dotenv

# Features
Automatic language detection (Arabic / English)
Structured JSON output
Sarcasm and mixed sentiment handling
JSONL result logging for experiments

# Example
### Input:
"I absolutely loved the service… until it crashed."
### Output:
{
  "language": "en",
  "sentiment": "mixed"
}
