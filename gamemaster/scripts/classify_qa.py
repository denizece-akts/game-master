import json
import re
from pathlib import Path

QA_PATH = Path("gamemaster/resources/qa_v3.jsonl")

def classify_item(item):
    query = item["query"].lower()
    history = item.get("history", [])
    
    # 1. Follow-up: Has history
    if len(history) > 0:
        return "follow-up"
    
    # 2. Comparison: specific keywords
    # Matches: "how does x differ", "compare x and y", "difference between", "x vs y", "distinguishes", "separates"
    if any(k in query for k in ["differ", "compare", "difference", " vs ", "distinguishes", "separates"]):
        return "comparison"
        
    # 3. Default to Factual
    return "factual"

def main():
    if not QA_PATH.exists():
        print(f"Error: {QA_PATH} not found.")
        return

    print(f"Processing {QA_PATH}...")
    
    new_items = []
    counts = {"factual": 0, "comparison": 0, "follow-up": 0}
    
    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q_type = classify_item(item)
            item["type"] = q_type
            counts[q_type] += 1
            new_items.append(item)
            
    # Write back to file
    with open(QA_PATH, "w", encoding="utf-8") as f:
        for item in new_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Done!")
    print("Counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
