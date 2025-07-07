import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device set to ', device)
df = pd.read_csv('cleaned_interview_questions.csv', engine='python', encoding='latin1', on_bad_lines='warn')
df['text'] = df['Company'] + '. ' + df['cleaned_question']
documents = [
    Document(page_content=row['text'], metadata={"name": row['Company'], "questions": row.get('cleaned_question', 'N/A')})
    for _, row in df.iterrows()
]
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device})

db = FAISS.from_documents(documents, embedding=embedder)
db.save_local("faiss_index/company_questions")
print("FAISS index saved to faiss_index/company_questions")