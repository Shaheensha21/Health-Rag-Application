# 💊 Health RAG Application

An AI-powered **Health Question Answering System** built using **Retrieval-Augmented Generation (RAG)**.  
This application retrieves relevant health information from curated documents and generates concise, context-aware answers using a Large Language Model.

🔗 **Live Demo:**  
https://health-rag-application.streamlit.app/

---

## 🧠 Project Overview

The Health RAG Application combines **semantic search** and **generative AI** to answer health-related questions accurately and responsibly.  
Instead of relying only on an LLM, it first retrieves relevant information from a knowledge base and then generates grounded responses.

---

## ✨ Key Features

- Retrieval-Augmented Generation (RAG) architecture  
- Semantic search using vector embeddings  
- Google Gemini LLM integration  
- Secure API key handling (no hardcoded secrets)  
- Clean, professional Streamlit UI  
- Cached vector database for fast performance  
- Medical disclaimer for responsible usage  

---

## 🏗️ Architecture

1. User enters a health-related query  
2. Query is converted into embeddings  
3. Relevant documents are retrieved from ChromaDB  
4. Retrieved context is passed to Google Gemini  
5. Concise and context-aware answer is generated  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Frontend:** Streamlit  
- **LLM:** Google Gemini  
- **Framework:** LangChain  
- **Vector Database:** ChromaDB  
- **Embeddings:** HuggingFace Sentence Transformers  

---

## 📂 Project Structure
Health-RAG-Application/
│── app.py # Streamlit application
│── data/ # Health documents
│── notebooks/ # Experiments & preprocessing
│── health_rag_db/ # Persisted vector database
│── requirements.txt # Dependencies
│── README.md # Project documentation


---

## 🔐 Environment Variables

Create a `.env` file locally or add secrets in Streamlit Cloud:


⚠️ Never commit your API key to GitHub.

---

## ▶️ How to Run Locally

git clone https://github.com/your-username/Health-RAG-Application.git
cd Health-RAG-Application
pip install -r requirements.txt
streamlit run app.py

👤 Author

Shaik Abdul Shahansha
🎓 MCA Student | Data & AI Enthusiast | Machine Learning Explorer

📫 Feel free to connect and collaborate!



