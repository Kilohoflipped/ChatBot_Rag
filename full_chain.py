import os

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from basic_chain import get_model
from filter import ensemble_retriever_from_docs
from local_loader import load_txt_files
from memory import create_memory_chain
from rag_chain import make_rag_chain


def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model(openai_api_key=openai_api_key)
    system_prompt = """你是一位专门为西北工业大学(NWPU)学生提供咨询的 AI 助手。
    请结合下文提供的教务管理规定、选课办法、GPA计算规则及校园生活指南, 以及用户的聊天记录来回答问题。
    
    要求：
    1. 必须基于提供的上下文(Context)进行回答。
    2. 如果上下文中没有相关信息，请直接回答“抱歉，根据目前的文档库我无法回答该问题”，不要编造政策。
    3. 回答语调应专业、严谨且富有帮助。
    4. 回答内无需指出具体资料的名字，直接说“根据资料库”即可
    
    Context: {context}
    
    Question: """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query}, config={"configurable": {"session_id": "foo"}}
    )
    return response
