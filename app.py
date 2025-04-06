from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import torch
import logging


logging.basicConfig(
    filename="mindly_debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


app = Flask(__name__)
load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")


try:
    embeddings = download_hugging_face_embeddings()
    index_name = "mindly"
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logging.info("Successfully initialized Pinecone retriever.")
except Exception as e:
    logging.error(f"Error initializing Pinecone retriever: {str(e)}")


try:
    llm = OllamaLLM(
        model="mistral:7b",
        temperature=0.4,
        top_p=0.9,
        max_tokens=300,
        repeat_penalty=1.1
    )
    logging.info("Loaded Mistral model with Ollama.")
except Exception as e:
    logging.error(f"Error loading LLM: {str(e)}")


system_prompt = (
    "You are Mindly, a kind and empathetic mental health assistant. "
    "Your responses should be helpful, calming, and clearly formatted. "
    "Always follow this format:\n"
    "1. Short point 1\n"
    "2. Short point 2\n"
    "...\n"
    "Limit your reply to a maximum of **5 numbered points**. Each point should be no longer than 1 sentence. "
    "Never write long paragraphs or more than 5 points. "
    "If the context doesn’t help, kindly say: “I'm here for you, but I may need more information to help with that.” "
    "Do not mention 'context' or model limitations.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


embedder = SentenceTransformer("all-MiniLM-L6-v2")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        logging.info(f"User input: {msg}")
        print("User:", msg)

        query_embedding = embedder.encode(msg, convert_to_tensor=True)
        retrieved_docs = retriever.invoke(msg)

        relevant_docs = []
        for doc in retrieved_docs:
            doc_embedding = embedder.encode(doc.page_content, convert_to_tensor=True)
            similarity = util.cos_sim(query_embedding, doc_embedding).item()
            logging.info(f"Doc snippet: {doc.page_content[:100]}... | Similarity: {similarity:.4f}")
            if similarity > 0.47:
                relevant_docs.append(doc)

        if not relevant_docs:
            response_text = "I'm here to support you, but I may need a bit more information to assist you effectively."
            logging.info("No relevant docs found. Sent fallback response.")
        else:
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": msg})
            response_text = response["answer"]
            logging.info(f"Generated response: {response_text}")

        print("Response:", response_text)
        return str(response_text)

    except Exception as e:
        logging.error(f"Error during chat flow: {str(e)}")
        return "Sorry, something went wrong. Please try again."



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
