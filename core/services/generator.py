import cohere
import os

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

def generate_answer_stream(query, context, strict=False):

    if strict:
        system_prompt = f"""
You are a QA system for Telkom University.

STRICT RULES:
- Answer ONLY from the provided context.
- Do NOT use external knowledge.
- If answer not found, reply EXACTLY with:
"Maaf, informasi tersebut tidak tersedia dalam data yang saya miliki."

Context:
{context}
"""
    else:
        system_prompt = f"""
You are a helpful assistant about Telkom University Campus.
Answer using context below.

Context:
{context}
"""

    try:
        stream = co.chat_stream(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.8,
        )

        for event in stream:

            if event.type == "content-delta":
                try:
                    text = event.delta.message.content.text
                    if text:
                        yield text
                except Exception:
                    pass
                
    except Exception as e:
        yield f"\n[STREAM ERROR] {str(e)}"