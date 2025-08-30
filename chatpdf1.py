import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings #inbuilt for vector embedding
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", 
    don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)#automatically converts question to vector before comparing
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    #automatically reads docs and its text is convertedto chain as context..its smart enough
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config(
        page_title="Chat PDF with Gemini",
        page_icon="💁",
        layout="wide"
    )

    st.title("📄 Chat with Your PDFs")
    st.subheader("Upload PDFs and ask questions from their content using Gemini LLM!")

    # User question input
    user_question = st.text_input("💬 Ask a Question from your PDF Files:")

    if user_question:
        user_input(user_question)

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📂 PDF Upload Menu")
        st.write("1️⃣ Upload one or more PDF files")
        st.write("2️⃣ Click 'Process PDFs' to extract text and prepare for Q&A")
        
        pdf_docs = st.file_uploader(
            "Upload your PDF Files here",
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
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed successfully!")



if __name__ == "__main__":
    main()
