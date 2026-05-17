import cohere
import os

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

def generate_answer_stream(query, context, strict=False):

    if strict:
        system_prompt = f"""
You are a QA system for Telkom University.
Answer ONLY if the question is about Telkom University Campus, otherwise say "Maaf, informasi tersebut tidak tersedia dalam data yang saya miliki."
Answer using context below.

STRICT RULES:
- Answer ONLY from the provided context.
- Do NOT use external knowledge.
- If answer not found, reply EXACTLY with:
  "Maaf, informasi tersebut tidak tersedia dalam data yang saya miliki."
- For ALL mathematical expressions, formulas, and symbols — ALWAYS use LaTeX notation:
  * Use \\( ... \\) for inline math (e.g. \\( \\frac{{N}}{{N_m}} \\times 4 \\))
  * Use \\[ ... \\] for block/display math
  * Use \\frac{{a}}{{b}} for division (NEVER use "/")
  * Use \\times for multiplication (NEVER use "x" or "*")
  * Use \\sqrt{{x}} for square roots (NEVER use "sqrt()")
  * Use subscripts like N_{{m}} with underscore
  * NEVER write math as plain text

Context:
{context}
"""
    else:
        system_prompt = f"""
You are a helpful assistant ONLY ANSWER about Telkom University Campus.
Answer using context below.

Context:
{context}
"""
        
# - If there are mathematical calculations, show the mathematical symbols and numbers as they are, do NOT convert them into words.


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