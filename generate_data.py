import os
import json
import time
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Run: $env:GROQ_API_KEY='your_key'")

client = Groq(api_key=api_key)

SYSTEM_PROMPT = """You are a machine learning data synthesis engine.
Generate training pairs for an extractive context compaction model.
The model scores whether a historical conversation sentence is essential to answer a user's latest query.

Output a valid JSON object matching this schema:
{
  "data": [
    {
      "query": "string",
      "sentence": "string",
      "label": 1.0 or 0.0
    }
  ]
}

- 'label' must be 1.0 for essential technical facts, code, configurations, or direct answers.
- 'label' must be 0.0 for conversational filler, greetings, pleasantries, or off-topic tangents.
"""

DOMAINS = [
    "database timeouts, connection pools, and query optimization",
    "FastAPI, async workers, and microservice architectures",
    "git merge conflicts, rebasing, and branch workflows",
    "Docker containers, memory limits, and OOM issues",
    "machine learning pipelines, ONNX quantization, and tokenizers"
]

def generate_sample_batch(domain: str, count: int = 20) -> list[dict]:
    prompt = f"""Topic: {domain}
Generate {count} balanced query-sentence pairs (10 with label 1.0, 10 with label 0.0).
Ensure the root output is a JSON object with key "data" containing the list of items.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        raw_text = response.choices[0].message.content.strip()
        parsed = json.loads(raw_text)

        # Handle both list and dict structures defensively
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            if "data" in parsed and isinstance(parsed["data"], list):
                items = parsed["data"]
            elif "pairs" in parsed and isinstance(parsed["pairs"], list):
                items = parsed["pairs"]
            else:
                # Fallback to the first list found in values
                items = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            items = []

        valid_samples = []
        for item in items:
            if isinstance(item, dict) and "query" in item and "sentence" in item and "label" in item:
                valid_samples.append({
                    "query": str(item["query"]).strip(),
                    "sentence": str(item["sentence"]).strip(),
                    "label": float(item["label"])
                })

        return valid_samples

    except Exception as e:
        print(f"[!] Batch error on domain '{domain}': {e}")
        return []

def main():
    all_samples = []
    print("[*] Starting compaction dataset generation...")

    for domain in DOMAINS:
        print(f"[*] Synthesizing samples for: {domain}")
        batch = generate_sample_batch(domain, count=20)
        print(f"    --> Collected {len(batch)} valid samples.")
        all_samples.extend(batch)
        time.sleep(1.2)  # Prevent hitting rate limits

    # Deduplicate entries
    unique_map = {(s["query"], s["sentence"]): s for s in all_samples}
    dataset = list(unique_map.values())

    if not dataset:
        print("[!] No samples were generated. Check your GROQ_API_KEY and connection.")
        return

    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open("test.jsonl", "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n[SUCCESS] Generated dataset ready!")
    print(f" - train.jsonl: {len(train_data)} rows")
    print(f" - test.jsonl:  {len(test_data)} rows")

if __name__ == "__main__":
    main()