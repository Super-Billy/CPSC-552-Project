import os
from pathlib import Path

def merge_jsonl_files(dir_path: str) -> None:
    """
    Merge all .jsonl files (excluding combined.jsonl) in the given directory
    into a single file at dir_path/combined.jsonl.

    Args:
        dir_path (str): Target directory path.
    """
    dir_path = Path(dir_path)
    output_file = dir_path / "combined.jsonl"

    # Open the output file in write mode to overwrite or create it
    with output_file.open('w', encoding='utf-8') as out_f:
        # Iterate through all .jsonl files
        for jsonl_path in dir_path.glob("*.jsonl"):
            # Skip the output file itself
            if jsonl_path.name == output_file.name:
                continue
            # Read and write line by line
            with jsonl_path.open('r', encoding='utf-8') as in_f:
                for line in in_f:
                    out_f.write(line)
    
    print(f"Merging completed, written to: {output_file}")



import json

def increment_row_number(jsonl_path):
    """
    Read a JSONL file, add 1 to the 'Row Number' field in each JSON object,
    and overwrite the original file with the updated content.
    """
    updated_objects = []
    # Read and update
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "Row Number" in obj:
                value = obj["Row Number"]
                try:
                    num = int(value)
                    obj["Row Number"] = num - 1
                except (TypeError, ValueError):
                    # Skip if it's not a valid integer
                    pass
            updated_objects.append(obj)
    
    # Overwrite the original file
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for obj in updated_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')



def remove_error_entries(jsonl_path: str) -> None:
    """
    Read a JSONL file and remove entries where the "error" field is not None.
    Write the remaining entries back to the original file.

    Args:
        jsonl_path (str): Path to the .jsonl file to process.
    """
    kept_lines = []

    # Read and filter
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Keep line if JSON parsing fails
                kept_lines.append(line)
                continue

            # Keep the entry only if "error" is None or not present
            if obj.get("error") is None:
                kept_lines.append(line)

    # Overwrite the original file
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for ln in kept_lines:
            f.write(ln + '\n')

    print(f"Processing complete. Kept {len(kept_lines)} records and wrote them back to `{jsonl_path}`.")




from typing import List
def find_missing_and_errors(jsonl_path: str) -> List[int]:
    """
    Read a JSONL file and return a list that includes:
      1. Row Numbers missing in the range 0-1046
      2. Row Numbers where the "error" field is not None
    Also prints the length of the resulting list.

    Args:
        jsonl_path (str): Path to the input .jsonl file

    Returns:
        List[int]: Sorted list of row numbers
    """
    row_present = set()
    error_rows = set()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip lines that fail to parse
                continue

            row = obj.get("Row Number")
            if isinstance(row, int):
                row_present.add(row)
                # If "error" field exists and is not None, mark as error row
                if obj.get("error") is not None:
                    error_rows.add(row)

    # Full set of expected row numbers
    full_set = set(range(1, 1048))
    # Missing row numbers
    missing_rows = full_set - row_present
    # Combine missing rows and error rows
    result = sorted(missing_rows.union(error_rows))
    # Print the number of matching rows
    print(f"Total of {len(result)} matching row numbers.")
    return result