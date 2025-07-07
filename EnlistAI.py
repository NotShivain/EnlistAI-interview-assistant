import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
import os
import requests
import json
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
st.toast("✅ Environment variables loaded.")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
api_key = os.getenv("API_KEY")
st.toast("Groq configuration set")
st.toast("LangChain configuration set.")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.toast(f"🖥 Torch device set to: {device}")

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)
st.toast("🔎 HuggingFace Embeddings initialized.")

db = FAISS.load_local(
    "faiss_index/company_questions",
    embeddings=embedder,
    allow_dangerous_deserialization=True
)
st.toast("FAISS index loaded from local storage.")

def get_company_details_from_api(company_name):
    try:
        encoded_company_name = company_name.replace(" ", "%20")
        api_url = f"https://enlistai.onrender.com/company/{encoded_company_name}"
        
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            company_data = response.json()
            return company_data
        else:
            st.error(f"Failed to fetch company details. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching company details: {str(e)}")
        return None


template = """
You are an AI interview assistant. The user wants: {question}

Here are some relevant questions from various companies:
{context}

Based on these, suggest the best fitting questions and provide answers as per user's request.
"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template
)
st.toast("📋 PromptTemplate initialized.")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
st.toast("🤖 Ollama LLM initialized.")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)
st.toast("🔗 RetrievalQA chain initialized.")


st.title('EnlistAI - Your AI Interview Assistant')
company_name = st.text_input("Search the company you want")
input_text = st.text_input("Enter your query (e.g., 'What are the interview questions for Google?')")
if company_name:
    company_data = get_company_details_from_api(company_name)
    if company_data:
        st.write(f"**Company:** {company_data['Company']}")
        st.write(f"**Rating:** {company_data.get('Rating', 'N/A')}")
        st.write(f"**Reviews:** {company_data.get('Reviews', 'N/A')}")
        st.write(f"**Industry & Location:** {company_data.get('Industry & Location', 'N/A')}")
if input_text:
    response = rag_chain.run(input_text)
    
    st.write(response)
    st.toast("✅ Query processed by RAG chain.")
