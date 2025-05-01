from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os, logging

# ─── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="mindly_debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Flask & env ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# ─── Pinecone retriever ──────────────────────────────────────────────────────────
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="mindly",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ─── LLM & prompt ───────────────────────────────────────────────────────────────
llm = OllamaLLM(model="mistral:7b", temperature=0.4, top_p=0.9, max_tokens=300, repeat_penalty=1.1)
system_prompt = (
    "You are Mindly, a kind and empathetic mental health assistant. "
    "Your responses should be helpful, calming, and clearly formatted. "
    "Always follow this format:\n"
    "1. Short point 1\n"
    "2. Short point 2\n"
    "...\n"
    "Limit to **5 numbered points**, each ≤1 sentence. "
    "If you can’t help, say: “I'm here for you, but I may need more information to help with that.”\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ─── Embedder ────────────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Routes ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("chat2.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        logging.info(f"User input: {msg}")

        # 1) retrieve top-3
        docs = retriever.invoke(msg)

        # 2) compute similarities
        sims = []
        query_emb = embedder.encode(msg, convert_to_tensor=True)
        for d in docs:
            sim = util.cos_sim(query_emb, embedder.encode(d.page_content, convert_to_tensor=True)).item()
            sims.append(sim)
            logging.info(f"Doc sim={sim:.4f} | {d.page_content[:60]!r}")

        # 3) average-similarity gate
        avg_sim = sum(sims) / len(sims)
        logging.info(f"Average sim = {avg_sim:.4f}")
        if avg_sim < 0.45:
            logging.info("Average below threshold → fallback")
            return "I'm here for you, but I may need more information to help with that."

        # 4) else, call the LLM with all retrieved docs as context
        chain = create_stuff_documents_chain(llm, prompt)
        out = chain.invoke({"input": msg, "context": docs})
        answer = out if isinstance(out, str) else out.get("answer")
        logging.info(f"Generated answer: {answer!r}")
        return answer

    except Exception as e:
        logging.error(f"Chat error: {e}", exc_info=True)
        return "Sorry, something went wrong. Please try again."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
