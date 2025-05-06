from langchain_openai import AzureChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from browser_use import BrowserConfig
from dotenv import load_dotenv
import os
from pydantic import SecretStr
load_dotenv()
import asyncio

from browser_use import Agent

load_dotenv()


llm = AzureChatOpenAI(
    model="gpt-4o",
    api_version='2024-02-15-preview',
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY") or ""),
)

config = BrowserConfig(
    headless=False,
    disable_security=True
)

browser = Browser()
context = BrowserContext(browser=browser, config=config)



task = 'What is the weather in NYC today?'

agent = Agent(task=task, llm=llm, browser_config=config)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
