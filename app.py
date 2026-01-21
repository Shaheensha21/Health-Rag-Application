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
# Custom CSS
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(120deg, #d0e7f9, #ffffff);
        color: #1f2937;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0b3d91;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    div.stTextInput>div>div>input {
        height: 50px;
        font-size: 1.1rem;
        border-radius: 10px;
        padding-left: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Title
# ---------------------------
st.markdown('<div class="title">💊 Health Chat Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask your health-related questions and get concise, professional answers</div>',
    unsafe_allow_html=True
)

# ---------------------------
# Load Vector DB
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
# Initialize Gemini LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY")
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
If the context does not contain relevant information, say:
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
question = st.text_input("Enter your health-related query:")

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
            st.error("❌ An error occurred while generating the response.")
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
    It is not a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider.
    </div>
    """,
    unsafe_allow_html=True
)
