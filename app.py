# app.py
import os
import streamlit as st

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
# UI Styling
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(120deg, #d0e7f9, #ffffff);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-size: 2.4rem;
        font-weight: bold;
        color: #0b3d91;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">💊 Health Chat Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask health questions and get concise, professional answers</div>',
    unsafe_allow_html=True
)

# ---------------------------
# Load Vector Store
# ---------------------------
with st.spinner("Loading health knowledge base..."):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory="health_rag_db",
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# Initialize Gemini LLM (CORRECT)
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    api_key=os.getenv("GEMINI_API_KEY")
)

# ---------------------------
# Prompt Template
# ---------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical information assistant.
Provide general health information only.
Do NOT provide diagnosis or prescriptions.

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
# Build RAG Chain
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
            answer_text = response.content

            for sentence in answer_text.split(". "):
                if sentence.strip():
                    st.markdown(f"- {sentence.strip()}.")

        except Exception as e:
            st.error("❌ Error while generating response")
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
