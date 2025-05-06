import os
import json
import asyncio
import tqdm
import argparse
import datetime
from langchain_openai import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
from browser_use.browser.browser import Browser, BrowserConfig
from helper import load_dataset, parse_ans, check_correctness, llm_output_answer
from concurrent.futures import ThreadPoolExecutor
# from uti

load_dotenv()

async def process_partition(df_partition, websites, deepseek_api_key, start_row, end_row, thread_id, output_dir, timestamp):
    """
    Each thread handles tasks for a specific data row range and outputs results to a separate JSONL file
    """
    # Set OPENAI_API_KEY environment variable
    os.environ["OPENAI_API_KEY"] = deepseek_api_key

    # Initialize LLM and browser (each thread creates its own)
    llm = ChatOpenAI(
        # model="deepseek-chat",
        model="deepseek-reasoner",
        base_url='https://api.deepseek.com',
        api_key=deepseek_api_key,
        temperature=0.1
    )

    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True,
            extra_chromium_args=['--window-size=2000,2000'],
        )
    )

    # Each thread outputs to a separate file, filename includes thread ID
    output_path = os.path.join(output_dir, f"deepseek_{timestamp}_thread{thread_id}.jsonl")
    with open(output_path, "w", encoding="utf-8") as outfile:
        for idx in tqdm.tqdm(range(start_row, end_row), desc=f"Thread {thread_id}"):
            try:
                row = df_partition.iloc[idx]
                patient_note = row["Patient Note"]
                question = row["Question"]
                calculator_id = str(row["Calculator ID"])
                note_id = str(row["Note ID"])
                gt = row["Ground Truth Answer"]
                upper_limit = row["Upper Limit"]
                lower_limit = row["Lower Limit"]

                # Get corresponding URL from websites.json using calculator_id
                url = websites.get(calculator_id)
                if not url:
                    print(f"Row {idx}: URL for Calculator ID {calculator_id} not found in websites.json, skipping this entry")
                    continue

                # Construct task instruction
                task_instruction = f"""
                Task: Given the English patient note and the medical question, you are required to first understand the question and all relevant academic knowledge, then use the medical calculator from the specified website to compute the answer. You must analyze the structure of the website and provide correct interaction steps to complete the task.

                You should not rush to complete the task by giving too many steps at once. Your top priority is to progress step-by-step and ensure every operation instruction is correct.

                Here is the medical calculator URL: {url}
                Here is the patient note: {patient_note}
                Here is the question: {question}

                Your final output must be a valid JSON object in English, strictly adhering to the required JSON format. Ensure the output only contains raw JSON without any extra text.
                """

                # Create a new browser context for each task
                async with await browser.new_context() as context:
                    try:
                        agent = Agent(
                            task=task_instruction,
                            llm=llm,
                            browser_context=context,
                            use_vision=False
                        )
                        result = await agent.run(max_steps=10)
                        ans = parse_ans(result)
                    except Exception as inner_e:
                        raise Exception(f"Error occurred during browser interaction: {inner_e}")
                
                ans = llm_output_answer(ans)
                correctness = check_correctness(ans, gt, calculator_id, upper_limit, lower_limit)

                # Print basic info
                print(f"Row: {idx}, Calculator ID: {calculator_id}, URL: {url}, ans: {ans}, Ground Truth: {gt}")

                # Construct entry to save to JSON file
                entry = {
                    "Row Number": idx,
                    "Calculator ID": calculator_id,
                    "Category": row["Category"],
                    "Note ID": note_id,
                    "Patient Note": patient_note,
                    "Question": question,
                    "ans": ans,
                    "Ground Truth Answer": gt,
                    "upperlimit": upper_limit,
                    "lower limit": lower_limit,
                    "correctness": correctness,
                    "error": None
                }
            except Exception as e:
                print(f"Error in row {idx}: {e}")
                entry = {
                    "Row Number": idx,
                    "error": str(e)
                }
            # Write each entry as a JSON line
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # Close the browser after the loop
    await browser.close()

def thread_worker(df_partition, websites, deepseek_api_key, start_row, end_row, thread_id, output_dir, timestamp):
    """
    Each thread calls asyncio.run to execute the async task
    """
    asyncio.run(process_partition(df_partition, websites, deepseek_api_key, start_row, end_row, thread_id, output_dir, timestamp))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data processing task over specified row range, multithreaded version")
    parser.add_argument("--start_row", "-s", type=int, default=0, help="Start row index (inclusive), default is 0")
    parser.add_argument("--end_row", type=int, default=None, help="End row index (exclusive), default is end of dataset")
    parser.add_argument("--thread_count", "-t", type=int, default=1, help="Number of threads, default is 1")
    args = parser.parse_args()

    output_dir = "json_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("Please set DEEPSEEK_API_KEY in your .env file")
    
    # Load dataset and websites.json
    df_partition = load_dataset("data/test_data.csv")
    if args.end_row is None:
        args.end_row = len(df_partition)
    
    with open("data/websites.json", "r", encoding="utf-8") as f:
        websites = json.load(f)

    # Calculate total rows and rows per thread
    total_rows = args.end_row - args.start_row
    per_thread = total_rows // args.thread_count

    # Use a unified timestamp for all thread outputs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Launch multiple threads using ThreadPoolExecutor; each runs an async task
    with ThreadPoolExecutor(max_workers=args.thread_count) as executor:
        futures = []
        for i in range(args.thread_count):
            part_start = args.start_row + i * per_thread
            # Last thread handles all remaining rows
            if i == args.thread_count - 1:
                part_end = args.end_row
            else:
                part_end = part_start + per_thread
            futures.append(executor.submit(thread_worker, df_partition, websites, deepseek_api_key, part_start, part_end, i, output_dir, timestamp))
        # Wait for all threads to finish
        for future in futures:
            future.result()
