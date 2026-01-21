# app.py
import os
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="💊 Health Chat Assistant",
    page_icon="💊",
    layout="centered"
)

# ---------------------------
# UI
# ---------------------------
st.markdown("<h1 style='text-align:center;'>💊 Health Chat Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Ask health-related questions and get concise answers</p>",
    unsafe_allow_html=True
)

# ---------------------------
# Load Vector DB
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
# Initialize Gemini (SAFE WAY)
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)
# 🔐 API key is read automatically from environment (Streamlit Secrets)

# ---------------------------
# Prompt
# ---------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical information assistant.
Provide general health information only.
Do NOT give diagnosis or prescriptions.

Context:
{context}

Question:
{question}

Answer in no more than 3 sentences.
If the context is insufficient, say:
"I'm sorry, I cannot provide an answer based on the available health documents."
"""
)

# ---------------------------
# RAG function (CORRECT)
# ---------------------------
def run_rag(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    return llm.invoke(formatted_prompt)

# ---------------------------
# User Input
# ---------------------------
question = st.text_input("Enter your health-related question:")

if question:
    with st.spinner("Generating answer..."):
        try:
            response = run_rag(question)

            st.markdown("### ✅ Answer")
            for sentence in response.content.split(". "):
                if sentence.strip():
                    st.markdown(f"- {sentence.strip()}.")

        except Exception as e:
            st.error("❌ Error generating response")
            st.exception(e)

# ---------------------------
# Disclaimer
# ---------------------------
st.markdown(
    """
    <div style="
        font-size: 16px;
        font-weight: bold;
        color: #7f1d1d;
        background-color: #fee2e2;
        padding: 12px;
        border-radius: 10px;
        margin-top: 20px;
    ">
    ⚠️ Disclaimer: This chatbot provides general health information only.
    It is not a substitute for professional medical advice.
    Always consult a qualified healthcare provider.
    </div>
    """,
    unsafe_allow_html=True
)
