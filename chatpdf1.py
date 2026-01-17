import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


# Load env for local use
load_dotenv()

# Prefer Streamlit Secrets on Cloud, fallback to .env locally
api_key = None
try:
    api_key = st.secrets["COHERE_API_KEY"]
except Exception:
    api_key = os.getenv("COHERE_API_KEY")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context. 
If the answer is not in the context, say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatCohere(
        model="command-r-plus-08-2024",
        temperature=0.3,
        cohere_api_key=api_key
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=api_key
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with PDF (Cohere)", page_icon="📄", layout="wide")
    st.title("📄 Chat with Your PDFs (Cohere)")
    st.subheader("Upload PDFs and ask questions from their content")

    if not api_key:
        st.error("COHERE_API_KEY not found. Add it in Streamlit Secrets or .env file.")
        st.stop()

    user_question = st.text_input("💬 Ask a question from your PDF:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("📂 PDF Upload Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF files here",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("🚀 Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first!")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ PDFs processed successfully!")
                    else:
                        st.warning("Could not extract text from the uploaded PDF(s).")


if __name__ == "__main__":
    main()
