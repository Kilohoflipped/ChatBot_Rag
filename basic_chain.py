import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-reasoner"


def get_model(openai_api_key=None, **kwargs):
    api_key = openai_api_key or os.environ.get("DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model_name=DEEPSEEK_CHAT_MODEL,
        openai_api_base=DEEPSEEK_BASE_URL,
        openai_api_key=api_key,
        temperature=0,
    )


def basic_chain(model=None, prompt=None):
    if not model:
        model = get_model()
    if not prompt:
        prompt = ChatPromptTemplate.from_template(
            "Tell me the most noteworthy books by the author {author}"
        )

    chain = prompt | model
    return chain
