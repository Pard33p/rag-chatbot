from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def create_vectorstore(pdf_path):
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    if not docs:
        raise ValueError("No documents loaded from PDF. Check the file path or content.")

    embeddings = OpenAIEmbeddings()
    print("Embedding the documents...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Vectorstore created.")
    return vectorstore

def query_rag(vectorstore: FAISS, question: str) -> str:
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI()
    rersponse = llm.predict(f"Context: {context}\n\nQuestion: {question}")
    return rersponse
