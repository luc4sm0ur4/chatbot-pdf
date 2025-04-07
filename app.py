import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Página
st.title("🤖 Chatbot PDF")
st.write("Faça perguntas sobre o conteúdo dos seus PDFs!")

# Upload
uploaded_file = st.file_uploader("Faça upload de um PDF", type="pdf")
if uploaded_file:
    with open(f"pdfs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Carregar e processar PDF
    loader = PyPDFLoader(f"pdfs/{uploaded_file.name}")
    docs = loader.load_and_split()

    # Embeddings e base vetorial
    embeddings = OpenAIEmbeddings()  # Use a API key via env: OPENAI_API_KEY
    db = FAISS.from_documents(docs, embeddings) # Cria a base vetorial FAISS
    st.write("📚 PDF carregado e processado com sucesso!")  


    # Chat
    question = st.text_input("Sua pergunta:") 
    if question:
        chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff") # Cria o modelo de chat 
        docs_similares = db.similarity_search(question) # Busca os documentos mais similares
        st.write("🔍 Buscando documentos similares...") 
        resposta = chain.run(input_documents=docs_similares, question=question) # Gera a resposta
        st.write("🤖", resposta) 
        st.write("🔍 Documentos utilizados:")
        for doc in docs_similares:
            st.write(doc.metadata["source"]) # Mostra os documentos utilizados na resposta 
        st.write("🔍 Resumo dos documentos utilizados:") 
        for doc in docs_similares:
            st.write(doc.page_content) # Mostra o conteúdo dos documentos utilizados na resposta