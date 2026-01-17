import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

# Configure Cohere API
# This will use the COHERE_API_KEY from your .env file
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    st.error("Cohere API Key not found. Please add it to your .env file.")

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


# Create FAISS vector store using Cohere embeddings
def get_vector_store(text_chunks):
    if not api_key:
        st.error("Cannot create vector store without a Cohere API Key.")
        return None
    try:
        embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None


# Setup LLM for Q&A using Cohere
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
    # Using the newer 'command-r-plus-08-2024' model
    model = ChatCohere(
        model="command-r-plus-08-2024",
        temperature=0.3,
        cohere_api_key=api_key
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Handle user queries
def user_input(user_question):
    if not api_key:
        st.error("Cannot process question without a Cohere API Key.")
        return
        
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=api_key)
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    except FileNotFoundError:
        st.warning("Please process your PDF documents first before asking a question.")
    except Exception as e:
        st.error(f"An error occurred while getting the answer: {e}")


# Streamlit UI
def main():
    st.set_page_config(page_title="Legal Bot with Cohere", page_icon="💁", layout="wide")
    st.title("📄 Chat with Your Legal Documents")
    st.subheader("Upload PDFs and ask questions from their content using Cohere AI!")

    user_question = st.text_input("💬 Ask a Question from your documents Files:")
    if user_question:
        user_input(user_question)

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📂 PDF Upload Menu")
        st.write("1️⃣ Upload one or more PDF files")
        st.write("2️⃣ Click 'Process PDFs' to prepare for Q&A")

        pdf_docs = st.file_uploader(
            "Upload your Legal Documents Files here",
            type=["pdf"],
            accept_multiple_files=True
        )

        if pdf_docs:
            st.write("Uploaded PDFs:")
            for pdf in pdf_docs:
                st.write(f"- {pdf.name}")

        if st.button("🚀 Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first!")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ PDFs processed successfully!")
                    else:
                        st.warning("Could not extract text from the uploaded PDF(s).")


if __name__ == "__main__":
    main()

