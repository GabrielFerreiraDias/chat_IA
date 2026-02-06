import os
from pathlib import Path

import streamlit as st

from document_processador import DocumentProcessor
from agent import AIAgent


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "aprendizado"
MODEL_PATH = BASE_DIR / "modelos" / "model.pkl"


@st.cache_resource
def build_agent(use_llm: bool) -> AIAgent:
    dp = DocumentProcessor(str(DOCS_DIR))
    dp.process_all_documents()
    agent = AIAgent(dp, model_path=str(MODEL_PATH), use_llm=use_llm)
    if not MODEL_PATH.exists():
        agent.train_and_save_model()
    return agent


st.set_page_config(page_title="Chat IA - Melhoria Continua", page_icon="\U0001F9E0")

st.title("Chat IA - Melhoria Continua")

if "use_llm" not in st.session_state:
    st.session_state.use_llm = os.getenv("USE_LLM", "true").lower() == "true"

use_llm = st.session_state.use_llm
if use_llm:
    st.warning("LLM local ativo: respostas mais naturais, porem mais lentas.")
else:
    st.info("Modo tradicional ativo: respostas mais rapidas.")

with st.sidebar:
    st.subheader("Controles")
    toggle = st.toggle("Ativar LLM local", value=use_llm)
    if toggle != st.session_state.use_llm:
        st.session_state.use_llm = toggle
        st.cache_resource.clear()
        st.rerun()

    st.write("Status: LLM ativo" if use_llm else "Status: LLM desativado")
    st.caption(f"Docs: {DOCS_DIR}")
    st.caption(f"Modelo: {MODEL_PATH}")

    if st.button("Retreinar modelo"):
        agent = build_agent(use_llm)
        agent.train_and_save_model()
        st.success("Modelo retreinado com sucesso!")

    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

agent = build_agent(use_llm)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Digite sua pergunta")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            reply = agent.chat(prompt)
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
