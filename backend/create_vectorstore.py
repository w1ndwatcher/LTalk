import pandas as pd
import os
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document  # Updated import for Document schema
from dotenv import load_dotenv
from huggingface_hub import login

# === Config ===
EXCEL_FILE = "learntrail_qa.xlsx"
INDEX_DIR = "vectorstore/learntrail_faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
load_dotenv()
#HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# === Load Excel ===
df = pd.read_excel(EXCEL_FILE)
assert "question" in df.columns and "answer" in df.columns, "Excel must have 'question' and 'answer' columns"

# === Prepare Documents ===
documents = []
for _, row in df.iterrows():
    question = str(row["question"]).strip()
    print("QUESTION: ", question)
    answer = str(row["answer"]).strip()
    print("ANSWER: ",answer)
    content = f"Q: {question}\nA: {answer}"
    print(content)
    documents.append(Document(page_content=content, metadata={"source": "learntrail"}))

# === Load Embeddings ===
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# === Create FAISS Vectorstore ===
vectorstore = FAISS.from_documents(documents, embedding_model)

# === Save Vectorstore ===
os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
vectorstore.save_local(INDEX_DIR)
print(f"✅ Vectorstore created and saved at {INDEX_DIR}")