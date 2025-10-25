from flask import Flask, request, jsonify, Response, stream_with_context
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv
from general_queries import general_queries
from flask_cors import CORS
from queue import Queue
import threading


# ========== Global Variables ==========
load_dotenv()
app = Flask(__name__)
CORS(app)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print("🔄 Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"} # or "cuda" if available
)
print("✅ Embedding model loaded.")

print("🔄 Loading FAISS vectorstore...")
vectorstore = FAISS.load_local("vectorstore/learntrail_faiss_index", embedding_model)
print("✅ Vectorstore loaded.")

# ========== Routes ==========
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "").strip()

    if not query:
        return jsonify({"error": "Question is required"}), 400

    # Handle general greetings or thanks
    for keyword, response in general_queries.items():
        if keyword in query.lower():
            return jsonify({"answer": response, "sources": []})

    # Streaming generator using a queue
    def generate():
        q = Queue()

        class FlaskStreamCallback(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):
                q.put(token)

            def on_llm_end(self, *args, **kwargs):
                q.put(None)  # Signal completion

        # Instantiate the LLM
        llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("LLAMA_GROQ_KEY"),
            temperature=0.7,
            streaming=True,
            callbacks=[FlaskStreamCallback()]
        )

        # RAG setup
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False,
        )

        # Run the QA chain in a thread so it doesn't block
        def run_chain():
            try:
                qa_chain.run(query)
            except Exception as e:
                q.put(f"[ERROR]: {str(e)}")
                q.put(None)

        threading.Thread(target=run_chain).start()

        # Yield tokens as they come in
        while True:
            token = q.get()
            if token is None:
                break
            yield token

    return Response(stream_with_context(generate()), content_type="text/plain")


@app.route("/health", methods=["GET"])
def health():
    return "✅ Backend is up and running."


if __name__ == "__main__":
    app.run(debug=True)