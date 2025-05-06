# CPSC-552-Project




# MedCalc-Agent: A Browser-Augmented System for Assisting LLMs with Medical Computation

MedCalc-Agent is a lightweight pipeline designed to enhance the numerical reasoning capabilities of Large Language Models (LLMs) in medical computation tasks, such as risk score calculation, dosage estimation, and fluid management.  
This system integrates retrieval, structured input extraction, and browser automation using real-world online calculators from [MDCalc](https://www.mdcalc.com).

---

## 🔧 Installation & Environment Setup

### 1. Install the Browser-Use Framework

MedCalc-Agent builds on the excellent `browser-use` framework for LLM-augmented browser control.  
Please follow the official setup instructions from the `browser-use` GitHub repository:  
👉 https://github.com/browser-use/browser-use

This includes setting up Chromium-based browser drivers and ensuring the Playwright environment is correctly initialized.

### 2. Set Up the Python Environment

Install project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
````

Additionally, create a `.env` file in the root directory to configure your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

These keys are required for LLM-based components of the pipeline.

---

## 🚀 Running the Pipeline

Once everything is configured, you can run the full MedCalc-Agent pipeline via:

```bash
python medcalc_agent.py
```

The agent will:

1. Select the appropriate MDCalc calculator.
2. Extract patient-specific inputs from the dataset.
3. Use a browser controller to automate the calculator interaction.
4. Parse and store the results in structured JSON format.

---

## 📁 Output & Dataset

The results will be saved to:

```
json_output/combined.jsonl
```

This output corresponds to test cases defined in:

```
data/test_data.csv
```

You can cross-reference these with evaluation results reported in our accompanying paper for reproducibility and verification.

---

## ⚠️ Troubleshooting

This project involves browser automation and interaction with external LLM APIs (e.g., OpenAI, Deepseek), which can occasionally result in:

* DOM element loading failures
* Browser timeouts
* JSON format mismatches
* API latency issues

If you encounter any unexpected errors or need help replicating our results, please don't hesitate to reach out:

📧 Contact: **[benlu.wang@yale.edu](mailto:benlu.wang@yale.edu)**
