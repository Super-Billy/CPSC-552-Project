import json

def compute_accuracy(jsonl_path: str) -> float:
    """
    Read a JSONL file, count the "correctness" field (1 or 0) for each record,
    and compute the overall accuracy (correct_count / total_count).

    Args:
        jsonl_path (str): Path to the input .jsonl file

    Returns:
        float: Accuracy value (between 0.0 and 1.0)
    """
    total = 0
    correct = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            val = obj.get("correctness")
            # Only include records where correctness is 0 or 1
            if val in (0, 1):
                total += 1
                if val == 1:
                    correct += 1

    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Total records: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    return accuracy
