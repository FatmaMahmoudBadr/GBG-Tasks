import os
import json
import time
from datetime import datetime
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class SentimentOutput(BaseModel):
    language: Literal["ar", "en"]
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    # confidence: float = Field(ge=0.0, le=1.0)
    explanation: dict

parser = PydanticOutputParser(pydantic_object=SentimentOutput)

llm_reasoning = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0.3
)

reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a bilingual sentiment analysis expert (Arabic & English).

Instructions:
- Analyze sentiment deeply (sarcasm, dialect, implicit emotion).
- Reason step-by-step internally (Hidden CoT).
- NEVER reveal chain-of-thought.
- Return ONLY valid JSON.
- Follow this format exactly:
{format_instructions}
"""),

    ("human", """
Example 1:
Text: "I waited 3 hours for support. Amazing service."

Output:
{{
  "language": "en",
  "sentiment": "negative",
  "explanation": {{
    "en": "The statement uses sarcasm to express dissatisfaction.",
    "ar": "الجملة تستخدم السخرية للتعبير عن عدم الرضا."
  }}
}}

Example 2:
Text: "الخدمة كويسة بس التأخير كان مستفز"

Output:
{{
  "language": "ar",
  "sentiment": "mixed",
  "explanation": {{
    "en": "Positive feedback is mixed with frustration about delay.",
    "ar": "يوجد رأي إيجابي عن الخدمة مع انزعاج من التأخير."
  }}
}}

Now analyze this text:
"{text}"
""")
])


reasoning_chain = reasoning_prompt | llm_reasoning | parser

def run_reasoning_sentiment(text: str):
    return reasoning_chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })

test_cases = [
    "I absolutely loved the service… until it crashed.", # mixed
    "Great job wasting my time.",                        # negative
    "The meeting is scheduled for 3 PM.",                # neutral
    "This is not bad at all.",                           # positive

    "الخدمة ممتازة ولكن التأخير غير مقبول.",          # mixed
     "الدنيا تمام بس التطبيق طلع يقفل فجأة.",         # mixed
    "يعني إيه خدمة عملاء؟ ولا حد بيرد.",                # negative
    "الخدمة كانت great بس السعر مبالغ فيه.",          # mixed
]

def run_test_suite(test_cases, output_file="E:\GBG\LangChain\langchain_tasks\sentiment_reasoning_results.jsonl"):
    results = []

    for idx, text in enumerate(test_cases, 1):
        try:
            output = run_reasoning_sentiment(text)

            record = {
                "input_text": text,
                "output": output.model_dump(),
            }

            results.append(record)

            print(f"[{idx}/{len(test_cases)}] Done")
        
        except Exception as e:
            print(f"[{idx}] Failed:", e)
        time.sleep(12)

    # Save as JSONL (best for experiments)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_file}")
run_test_suite(test_cases)


