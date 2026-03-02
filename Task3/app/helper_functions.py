def clean_sql(query: str) -> str:
    return query.replace("```sql", "").replace("```", "").strip()


def is_safe_query(query: str) -> bool:
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate"]
    return not any(word in query.lower() for word in forbidden)