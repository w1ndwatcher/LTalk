from flask import Flask, request, jsonify, Response, stream_with_context
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from general_queries import general_queries
from flask_cors import CORS
from queue import Queue
import threading
from huggingface_hub import login

# ========== Global Variables ==========
load_dotenv()
app = Flask(__name__)
CORS(app)

login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/learntrail_faiss_index")

print("🔄 Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"} # or "cuda" if available
)
print("✅ Embedding model loaded.")

print("🔄 Loading FAISS vectorstore...")
vectorstore = FAISS.load_local(
    "vectorstore/learntrail_faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ Vectorstore loaded.")

custom_prompt = PromptTemplate.from_template("""
You are LTalk, a friendly and knowledgeable AI assistant for LearnTrail — a platform that offers training programs, workshops, internships, and career support.

Always respond helpfully and clearly, even if the context doesn't directly mention the answer. Never say "the context provided does not mention". Instead, politely say you're not sure and invite the user to ask about LearnTrail’s services. 
Do not end your response with a trailing question.

Context:
{context}

Question:
{question}

Helpful answer:
""")


# ========== Routes ==========
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "").strip()

    if not query:
        return "Question is required"

    # Handle general greetings
    for keyword, response in general_queries.items():
        if keyword in query.lower():
            def general_response():
                yield response
            return Response(stream_with_context(general_response()), content_type="text/plain")

    # Step 1: Check similarity of top retrieved docs
    results = vectorstore.similarity_search_with_score(query, k=3)
    docs = [doc for doc, score in results if score > 0.7]
    if not docs:
        return jsonify({
            "answer": "❓ Sorry, I couldn't find relevant information for your question.",
            "sources": []
        })

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 2: Proceed with streaming response
    def generate():
        q = Queue(maxsize=20)
        stop_event = threading.Event()

        class FlaskStreamCallback(BaseCallbackHandler):
            def __init__(self, q, stop_event):
                self.q = q
                self.stop_event = stop_event
                
            def on_llm_new_token(self, token: str, **kwargs):
                if not self.stop_event.is_set():
                    self.q.put(token)
                
            def on_llm_error(self, error: Exception, **kwargs):
                self.q.put(f"\n[ERROR]: {str(error)}")
                self.q.put(None)
                
            def on_llm_end(self, *args, **kwargs):
                self.q.put(None)

        llm = ChatGroq(
            model_name=LLM_MODEL_NAME,
            api_key=os.getenv("LLAMA_GROQ_KEY"),
            temperature=0.7,
            streaming=True,
            callbacks=[FlaskStreamCallback(q, stop_event)],
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,  # show sources in the frontend
            chain_type_kwargs={"prompt": custom_prompt}
        )

        def run_chain():
            try:
                qa_chain.run(query)
            except Exception as e:
                print(f"Chain error: {str(e)}")
                q.put(f"[ERROR]: {str(e)}")
                q.put(None)

        thread = threading.Thread(target=run_chain, daemon=True)
        thread.start()

        try:
            while True:
                token = q.get()
                if token is None:
                    break
                yield token
        except GeneratorExit:
            # This block triggers if the client disconnects
            print("🔌 Client disconnected. Setting stop_event...")
            stop_event.set()
            thread.join(timeout=2)

    return Response(stream_with_context(generate()), content_type="text/plain")



@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "components": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_model": LLM_MODEL_NAME,
            "vectorstore": "loaded" if vectorstore else "unavailable"
        }
    })


if __name__ == "__main__":
    app.run(debug=True)