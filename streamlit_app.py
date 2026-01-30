import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever

from full_chain import create_full_chain, ask_question
from vector_store import DEFAULT_EMBEDDING_MODEL

STORE_DIR = "store"
CHROMA_COLLECTION = "chroma"
CHROMA_PERSIST_PATH = os.path.join(STORE_DIR, CHROMA_COLLECTION)
CHUNK_TEXTS_PATH = os.path.join(STORE_DIR, "chunk_texts.json")

st.set_page_config(page_title="学业小助手")
st.title("学业小助手")


def show_ui(qa, prompt_to_user="有什么可以帮您？"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = ask_question(qa, prompt)
                st.markdown(response.content)
        message = {"role": "assistant", "content": response.content}
        st.session_state.messages.append(message)


def _prebuilt_exists():
    return os.path.isdir(CHROMA_PERSIST_PATH) and os.path.isfile(CHUNK_TEXTS_PATH)


@st.cache_resource
def get_retriever():
    if not _prebuilt_exists():
        st.error(
            "未检测到已建好的向量库。请先在终端运行：`python build_db.py` 建库后再刷新本页。"
        )
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    db = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_PATH,
    )
    vs_retriever = db.as_retriever()
    with open(CHUNK_TEXTS_PATH, "r", encoding="utf-8") as f:
        chunk_texts = json.load(f)
    bm25_retriever = BM25Retriever.from_texts(chunk_texts)
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever], weights=[0.5, 0.5]
    )


def get_chain(deepseek_api_key=None):
    ensemble_retriever = get_retriever()
    chain = create_full_chain(
        ensemble_retriever,
        openai_api_key=deepseek_api_key,
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    )
    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("已找到API Key")
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"请输入您的 {secret_name}")
        secret_value = st.text_input(
            secret_name, key=f"input_{secret_key}", type="password"
        )
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[获取 {secret_name}]({info_link})")
    return secret_value


def run():
    deepseek_api_key = st.session_state.get("DEEPSEEK_API_KEY")

    with st.sidebar:
        if not deepseek_api_key:
            deepseek_api_key = get_secret_or_input(
                "DEEPSEEK_API_KEY",
                "DeepSeek API 密钥",
                info_link="https://platform.deepseek.com/api_keys",
            )

    if not deepseek_api_key:
        st.warning("缺少 DEEPSEEK_API_KEY")
        st.stop()

    chain = get_chain(deepseek_api_key=deepseek_api_key)
    st.subheader("向我提问本校学业、选课、GPA、考试等相关问题")
    show_ui(chain, "你好呀～选课、成绩、考试这些学业上的事都可以问我。")


run()
