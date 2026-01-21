# app.py
import streamlit as st
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="💊 Health Chat Assistant",
    page_icon="💊",
    layout="centered"
)

# ---------------------------
# Title
# ---------------------------
st.markdown("<h1 style='text-align:center;'>💊 Health Chat Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Ask health questions and get concise answers</p>",
    unsafe_allow_html=True
)

# ---------------------------
# Load Vector Store
# ---------------------------
with st.spinner("Loading knowledge base..."):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory="health_rag_db",
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# Initialize Gemini LLM (FINAL SAFE WAY)
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)
# ⚠️ NO api_key OR google_api_key PASSED

# ---------------------------
# Prompt
# ---------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical information assistant.
Provide general information only.
Do NOT diagnose or prescribe.

Context:
{context}

Question:
{question}

Answer in no more than 3 sentences.
If insufficient context, say:
"I'm sorry, I cannot provide an answer based on the available documents."
"""
)

# ---------------------------
# RAG Chain
# ---------------------------
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# ---------------------------
# User Input
# ---------------------------
question = st.text_input("Enter your health-related question:")

if question:
    with st.spinner("Generating answer..."):
        try:
            response = rag_chain.invoke({"question": question})

            st.markdown("### ✅ Answer")
            for s in response.content.split(". "):
                if s.strip():
                    st.markdown(f"- {s.strip()}.")

        except Exception as e:
            st.error("❌ Error generating response")
            st.exception(e)

# ---------------------------
# Disclaimer
# ---------------------------
st.markdown(
    """
    <div style="color:#7f1d1d; background:#fee2e2; padding:10px; border-radius:8px;">
    ⚠️ Disclaimer: This chatbot provides general health information only.
    It is not a substitute for professional medical advice.
    </div>
    """,
    unsafe_allow_html=True
)
