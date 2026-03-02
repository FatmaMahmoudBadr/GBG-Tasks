import os
import json
import time
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

parser = PydanticOutputParser(pydantic_object=SentimentOutput)

llm_reliable = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0
)

reliable_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a sentiment classification engine.

Rules:
1. Detect language (Arabic or English).
2. Choose ONE sentiment only:
   positive, negative, neutral, mixed.
3. If unclear → neutral.
4. Be conservative with confidence.
5. Output ONLY valid JSON.
Format:
{format_instructions}
"""),

    ("human", """
Text:
"{text}"

Steps (internal):
- Detect language
- Identify emotional polarity
- Assign sentiment
- Assign confidence
- Verify consistency

Return final JSON only.
""")
])

reliable_chain = reliable_prompt | llm_reliable | parser

def run_reliable_sentiment(text: str):
    return reliable_chain.invoke({
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

def run_test_suite(test_cases, output_file="E:\GBG\LangChain\langchain_tasks\sentiment_reliable_results.jsonl"):
    results = []

    for idx, text in enumerate(test_cases, 1):
        try:
            output = run_reliable_sentiment(text)

            record = {
                "input_text": text,
                "output": output.model_dump(),
            }

            results.append(record)

            print(f"✅ [{idx}/{len(test_cases)}] Done")
        
        except Exception as e:
            print(f"❌ [{idx}] Failed:", e)
        time.sleep(12)

    # Save as JSONL (best for experiments)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n📁 Results saved to {output_file}")
run_test_suite(test_cases)
