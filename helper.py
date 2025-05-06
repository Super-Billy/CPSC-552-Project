import re, math, json, os
import pandas as pd
import tqdm
from datetime import datetime
import numpy as np
import asyncio
from langchain_openai import ChatOpenAI


def check_correctness(answer: str, ground_truth, calid, upper_limit, lower_limit):
    upper_limit = str(upper_limit)
    lower_limit = str(lower_limit)

    calid = int(calid)
    if answer == "Not Found" or answer == "N/A":
        return 0

    if calid in [13, 68]:
        # Output Type: date
        if datetime.strptime(answer, "%m/%d/%Y").strftime("%-m/%-d/%Y") == datetime.strptime(ground_truth, "%m/%d/%Y").strftime("%-m/%-d/%Y"):
            correctness = 1
        else:
            correctness = 0
    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", ground_truth)
        ground_truth = f"({match.group(1)}, {match.group(3)})"
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f"({weeks}, {days})"
            if eval(answer) == eval(ground_truth):
                correctness = 1
            else:
                correctness = 0
        else:
            correctness = 0
    elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
        # Output Type: integer A
        answer = round(eval(answer))
        if answer == eval(ground_truth):
            correctness = 1
        else:
            correctness = 0
    elif calid in [2,  3,  5,  6,  7,  8,  9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:
        # Output Type: decimal
        answer = eval(answer)
        if answer >= eval(lower_limit) and answer <= eval(upper_limit):
            correctness = 1
        else:
            correctness = 0
    else:
        raise ValueError(f"Unknown calculator ID: {calid}")
    return correctness

def compute_overall_accuracy(output_path, model_name, prompt_style): 
    category_accuracy = {}

    with open(output_path) as file:
        for line in file:
            data = json.loads(line)
            
            category = data["Category"]

            if category not in category_accuracy:
                category_accuracy[category] = []

            if data["Result"] == "Correct":
                category_accuracy[category].append(1)
            else:
                category_accuracy[category].append(0)

    # Compute average and standard deviation for each category
    category_stats = {}
    all_results = []

    for cat, results in category_accuracy.items():
        results_array = np.array(results)
        category_mean = np.mean(results_array)
        category_std = round(np.sqrt(category_mean * (1-category_mean) / len(results_array)), 2)
        category_stats[cat] = {
            "average": round(category_mean * 100, 2),
            "std": category_std
        }
        all_results.extend(results)

    # Compute overall average and standard deviation
    all_results_array = np.array(all_results)
    overall_average = np.mean(all_results_array)
    overall_std =  round(np.sqrt(overall_average * (1-overall_average) / 1047), 2)

    category_stats["overall"] = {
        "average": round(overall_average * 100, 2),
        "std": overall_std
    }

    if not os.path.exists("stats"):
        os.makedirs("stats")

    if "/" in model_name:
        model_name = model_name.split('/')[1]

    with open(f"stats/results_{model_name}_{prompt_style}.json", "w") as file:
        json.dump(category_stats, file, indent=4)

    return category_stats

def extract_answer(answer, calid):
    calid = int(calid)
    
    # Try to directly parse the JSON data
    extracted_answer = answer
    
    # Post-process answer depending on calculator ID
    if calid in [13, 68]:
        # Output type: date, e.g. "01/15/2020"
        match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = match.group(3)
            answer_val = f"{month:02}/{day:02}/{year}"
        else:
            answer_val = "N/A"
    elif calid in [69]:
        # Output type: tuple (weeks, days)
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer_val = f"({weeks}, {days})"
        else:
            answer_val = "N/A"
    elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51]:
        # Output type: integer A
        match = re.search(r"(\d+) out of", extracted_answer)
        if match:
            answer_val = match.group(1)
        else:
            match = re.search(r"-?\d+(, ?-?\d+)+", extracted_answer)
            if match:
                answer_val = str(len(match.group(0).split(",")))
            else:
                match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                if len(match) > 0:
                    answer_val = match[-1][0]
                else:
                    answer_val = "N/A"
    elif calid in [2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:
        # Output type: decimal
        match = re.search(r"str\((.*)\)", extracted_answer)
        if match:
            expression = match.group(1)
            # Replace operators and function names for evaluation
            expression = (expression.replace("^", "**")
                                     .replace("is odd", "% 2 == 1")
                                     .replace("is even", "% 2 == 0")
                                     .replace("sqrt", "math.sqrt")
                                     .replace(".math", ""))
            # Remove comments
            expression = expression.split('#')[0]
            # Balance parentheses if needed
            if expression.count('(') > expression.count(')'):
                expression += ')' * (expression.count('(') - expression.count(')'))
            elif expression.count(')') > expression.count('('):
                expression = '(' * (expression.count(')') - expression.count('(')) + expression
            try:
                answer_val = eval(expression, {"__builtins__": None}, {
                    "min": min, "pow": pow, "round": round, "abs": abs,
                    "int": int, "float": float, "math": math, "np": np, "numpy": np
                })
            except Exception as e:
                print(f"Error in evaluating expression: {expression}\n{e}")
                answer_val = "N/A"
        else:
            match = re.search(r"(-?\d+(\.\d+)?)\s*mL/min/1.73", extracted_answer)
            if match:
                answer_val = eval(match.group(1))
            else:
                match = re.findall(r"(-?\d+(\.\d+)?)\%", extracted_answer)
                if len(match) > 0:
                    answer_val = eval(match[-1][0]) / 100
                else:
                    match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                    if len(match) > 0:
                        answer_val = eval(match[-1][0])
                    else:
                        answer_val = "N/A"
        if answer_val != "N/A":
            answer_val = str(answer_val)
    else:
        answer_val = extracted_answer
    
    if answer_val == "N/A":
        answer_val = "Not Found"

    return answer_val


def load_dataset(dataset_path):
    """
    Load CSV dataset and return as DataFrame.
    """
    print(f"[Main] Loading dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    print(f"[Main] Dataset loaded successfully, total {len(df)} rows found.")
    return df

def parse_ans(result):
    actions = result.model_actions()

    # Check if there are actions and take the last one
    if actions:
        last_action = actions[-1]
        # Get 'text' from 'done' dict in the last action
        if 'done' in last_action and 'text' in last_action['done']:
            text_value = last_action['done']['text']
        else:
            print("No 'done' or 'text' key in the last action")
    else:
        print("No actions retrieved")
    return str(text_value)


def llm_output_answer(ans):
    # Get API Key from environment variable. Make sure DEEPSEEK_API_KEY is set.
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("Please set the environment variable DEEPSEEK_API_KEY")

    # Initialize ChatOpenAI model with deepseek-reasoner
    llm = ChatOpenAI(
        model="deepseek-reasoner",
        base_url='https://api.deepseek.com',
        api_key=deepseek_api_key,
        temperature=0.6
    )

    # Define the prompt template; {result} will be replaced by actual content
    prompt_template = f"""
    For the text: {ans}
    This text is the output of a medical calculator. Please extract numeric or date information, as follows:
    - If the answer involves numeric calculation, identify the number representing the final result, and return only that number (without units or extra content).
    - If the answer involves a date, identify the final date and return it only, in the format MM/DD/YYYY (e.g., 08/31/2023 or 07/03/2000), with no other text.
    - If the answer describes age or duration (in the form of weeks and days), return a tuple clearly stating the weeks and days, e.g., (4 weeks, 3 days), (0 weeks, 5 days).
    """

    # Query the model and return the result
    response = llm(prompt_template)

    print(f"Original Answer: {ans} ------------------ Extracted Answer: {response.content} ------------------")

    return response.content
