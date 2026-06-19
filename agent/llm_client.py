import requests
import json


OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen2.5-coder:latest"


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_code(task_description: str, function_signature: str, hints: list[str] = None) -> str:
    system_prompt = (
        "You are an expert Python programmer. "
        "When asked to implement a function, respond with ONLY the complete Python function code. "
        "Do not include any explanation, markdown code fences, or extra text. "
        "Just the raw Python function definition."
    )

    user_content = f"Implement the following Python function:\n\n{function_signature}\n\n{task_description}"

    if hints:
        hint_block = "\n".join(f"- {h}" for h in hints)
        user_content += (
            f"\n\nFor reference, here are notes from similar past problems. "
            f"Apply them only if genuinely relevant to this specific problem — "
            f"ignore any that don't apply:\n{hint_block}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return chat(messages, temperature=1.2)


def analyze_failure(task_description: str, function_signature: str,
                    failed_code: str, test_failure_detail: str) -> dict:
    system_prompt = (
        "You are a code debugging expert. "
        "Analyze why the given code failed the test, then respond with a JSON object only — no markdown, no explanation. "
        "The JSON must have exactly these keys: "
        "\"error_category\" (short label like 'off-by-one' or 'boundary condition'), "
        "\"root_cause\" (one sentence explaining the bug), "
        "\"strategy_to_avoid\" (one concrete actionable tip to avoid this bug)."
    )

    user_content = (
        f"Task: {task_description}\n\n"
        f"Function signature: {function_signature}\n\n"
        f"Failed code:\n{failed_code}\n\n"
        f"Test failure: {test_failure_detail}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw = chat(messages, temperature=0.1)

    # Strip markdown fences if the model wraps in ```json ... ```
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract what we can
        return {
            "error_category": "unknown",
            "root_cause": raw[:200],
            "strategy_to_avoid": "Review the failed code carefully before retrying.",
        }
