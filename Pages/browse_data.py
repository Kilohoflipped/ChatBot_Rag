import streamlit as st

from local_loader import list_txt_files

paths = list(list_txt_files())

file_path = st.selectbox("选择要查看的数据文件", paths, index=None)

if file_path:
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        st.markdown(md_content)
