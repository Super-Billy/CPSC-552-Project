from langchain_openai import AzureChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from browser_use.agent.views import AgentHistoryList, ActionResult
from browser_use.controller.service import Controller
from browser_use.browser.context import BrowserContextConfig
from pydantic import SecretStr, BaseModel
from typing import Optional, List, Union
import asyncio
import os
from dotenv import load_dotenv
import json
load_dotenv()

# Initialize shared controller
controller = Controller()

class FormFieldOption(BaseModel):
    value: str
    label: str
    unit: Optional[str] = None

class FormField(BaseModel):
    name: str
    type: str  # 'text', 'radio', 'select', 'checkbox'
    description: str
    unit: Optional[str] = None  # For text inputs with units (e.g., mg, mL)
    options: Optional[List[FormFieldOption]] = None

class PageAnalysisResult(BaseModel):
    fields: dict[str, FormField]

@controller.action('analyze_medical_form', requires_browser=True)
async def analyze_medical_form(browser: Browser) -> ActionResult:
    """Analyzes medical calculator form structure and extracts field attributes"""
    page = browser.get_current_page()
    
    # Get all form elements and their containing labels/divs
    input_fields = await page.query_selector_all("input, select")
    analysis = {}
    
    for field in input_fields:
        # Get comprehensive field information including units
        field_info = await field.evaluate("""
            element => {
                const getFieldInfo = (el) => {
                    // Get label text
                    let label = '';
                    let unit = '';
                    
                    // Check for explicit label
                    if (el.labels && el.labels.length > 0) {
                        label = el.labels[0].textContent.trim();
                        
                        // Look for units in parentheses within label
                        const unitMatch = label.match(/\((.*?)\)/);
                        if (unitMatch) {
                            unit = unitMatch[1];
                            label = label.replace(/\s*\(.*?\)/, '').trim();
                        }
                    }
                    
                    // Check surrounding div for unit information
                    const parentDiv = el.closest('div');
                    if (parentDiv) {
                        const unitSpan = parentDiv.querySelector('.unit, [class*="unit"]');
                        if (unitSpan) {
                            unit = unit || unitSpan.textContent.trim();
                        }
                    }
                    
                    // Fallback to other attributes if no label found
                    if (!label) {
                        label = el.getAttribute('aria-label') || 
                               el.getAttribute('placeholder') || 
                               el.name || 
                               el.id;
                    }
                    
                    return {
                        name: el.name || el.id,
                        type: el.type || 'text',
                        label: label,
                        unit: unit,
                        isSelect: el.tagName.toLowerCase() === 'select'
                    };
                };

                return getFieldInfo(element);
            }
        """)

        field_type = field_info['type']
        field_options = None
        field_unit = field_info.get('unit', '')

        # Handle different input types
        if field_info['isSelect']:
            options = await field.evaluate("""
                element => Array.from(element.options).map(option => ({
                    value: option.value,
                    label: option.text.trim()
                }))
            """)
            field_type = 'select'
            field_options = [FormFieldOption(**opt) for opt in options]
        
        elif field_type in ['radio', 'checkbox']:
            same_name_fields = await page.query_selector_all(f"input[name='{field_info['name']}']")
            options = []
            for opt in same_name_fields:
                opt_info = await opt.evaluate("""
                    element => {
                        const label = element.labels?.[0]?.textContent.trim() || element.value;
                        const unitMatch = label.match(/\((.*?)\)/);
                        return {
                            value: element.value,
                            label: label.replace(/\s*\(.*?\)/, '').trim(),
                            unit: unitMatch ? unitMatch[1] : ''
                        };
                    }
                """)
                options.append(FormFieldOption(**opt_info))
            field_options = options

        analysis[field_info['name']] = FormField(
            name=field_info['name'],
            type=field_type,
            description=field_info['label'],
            unit=field_unit if field_type == 'text' else None,
            options=field_options
        )
    
    result = PageAnalysisResult(fields=analysis)
    
    return ActionResult(
        extracted_content=result.model_dump(),
        include_in_memory=True,
        is_done=True
    )

async def extract_medical_values(llm: AzureChatOpenAI, note: str, fields: dict[str, FormField]) -> dict:
    """Extracts values from patient note based on field descriptions"""
    field_descriptions = []
    for field_name, field in fields.items():
        if field.options:
            options_str = "\n    - " + "\n    - ".join(
                f"{opt.label} (value: {opt.value})" for opt in field.options
            )
            field_descriptions.append(f"""
{field_name}: {field.description}
Type: {field.type}
Available options: {options_str}
""")
        else:
            field_descriptions.append(f"""
{field_name}: {field.description}
Type: {field.type}
""")

    prompt = f"""
Extract the following values from the patient note. 
For fields with options, select the most appropriate option value.
For text fields, extract the relevant value.
Return a JSON object with field names as keys and extracted values.

Fields to extract:
{"".join(field_descriptions)}

Patient Note:
{note}
"""
    
    response = await llm.ainvoke(prompt)
    try:
        return json.loads(response.content)
    except:
        return {field: None for field in fields.keys()}

async def main():
    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True,
            extra_chromium_args=['--window-size=2000,2000'],
        )
    )

    llm = AzureChatOpenAI(
        model="gpt-4o",
        api_version='2024-02-15-preview',
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY")),
    )

    
    async with await browser.new_context() as context:
        try:
            analyzer = Agent(
                task="""Analyze this medical calculator page structure.
                       Extract all form fields and their attributes in JSON format.
                       For each field, identify and return:
                       1. Field name and description
                       2. Input type (text, radio, select, etc.)
                       3. Units for text inputs (mg, mL, etc.)
                       4. Available options for selection fields
                       DO NOT interact with the form - only observe and report.""",
                llm=llm,
                browser_context=context,
                controller=controller
            )

            url = "https://www.mdcalc.com/calc/2040/steroid-conversion-calculator"
            
            # Analyze the page structure
            analysis_result = await analyzer.run(
                task=f"Navigate to {url} and analyze the calculator form structure"
            )
            
            # Display the analysis results
            form_analysis = PageAnalysisResult(**analysis_result.final_result())
            
            print("\nForm Analysis:")
            for field_name, field in form_analysis.fields.items():
                print(f"\nField: {field_name}")
                print(f"Type: {field.type}")
                print(f"Description: {field.description}")
                if field.unit:
                    print(f"Unit: {field.unit}")
                if field.options:
                    print("Available Options:")
                    for opt in field.options:
                        option_text = f"  - {opt.label} (value: {opt.value})"
                        if opt.unit:
                            option_text += f" [{opt.unit}]"
                        print(option_text)

        finally:
            await browser.close()


        extractor_agent = AzureChatOpenAI(
            model="gpt-4o",
            api_version='2024-02-15-preview',
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY")),
        )

        messages = [{"system": "You are a helpful assistant that will extract values from a patient note. Please look at the keys of the JSON object and extract the values from the patient note.", "role": "user", "content": "Here is the patient note: {patient_note}. Here is the JSON of the form fields: {form_fields}."}]



if __name__ == "__main__":
    asyncio.run(main()) 

'''
# Analyzer Agent: Extracts field attributes from the page structure 
# - First, finds all the form fields needed in the website 
# - Determines if text, then provide units (numerical attribute). Otherwise, if field is a button, provide all the options. 
# 
# Return a JSON object for all inputs needed: {
#   field_name: {
#       type: str, 
#       description: str, 
#       unit: str, 
#       options: list[str]
#   }
# }



'''
