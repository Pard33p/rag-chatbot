# 🧠 RAG Chatbot (Retrieval-Augmented Generation)

This project is a simple RAG-based chatbot using FastAPI, LangChain, OpenAI, and FAISS. It allows users to ask questions based on the content of a PDF document by retrieving relevant text chunks and generating answers using OpenAI’s language model.

---

## 📁 Project Structure

- `app/`: Contains the FastAPI app and core RAG logic
  - `main.py`: Defines the API endpoints
  - `rag.py`: Handles PDF loading, chunking, embedding, and querying
- `data/`: Place your source PDF documents here
- `.env`: Stores the OpenAI API key
- `requirements.txt`: Lists Python dependencies
- `README.md`: Documentation

---

## 🚀 Features

- Loads and reads PDF documents
- Splits the document into manageable chunks
- Embeds the chunks using OpenAI's embedding model
- Stores the embeddings in an in-memory FAISS vector store
- Retrieves relevant chunks based on user queries
- Generates a context-aware answer using an OpenAI LLM
- Exposes a `/ask` endpoint for querying

---

## 🛠️ Setup Overview

1. **Install dependencies** using the provided `requirements.txt` file.
2. **Create a `.env` file** and set your OpenAI API key.
3. **Place your PDF** document inside the `data/` folder.
4. **Run the FastAPI app** with Uvicorn.

---

## 🧠 How It Works

- The PDF is loaded and split into smaller chunks of text.
- Each chunk is embedded using OpenAI’s embeddings API.
- These embeddings are stored in a FAISS vector store.
- When a query is made to the API, the system retrieves the most relevant chunks.
- The retrieved text is passed as context to the OpenAI chat model.
- The model generates a response based on the context and the question.

---

## 📤 Optional Enhancements

We can optionally extend this project to include:

- A file upload API for dynamic PDF ingestion
- A frontend using React or another framework
- Persistent vector store (e.g., saving FAISS to disk or using ChromaDB)
- Authentication or usage limits
- Logging and monitoring

---

## ✅ Use Cases

- Chatbots for internal company documents
- Educational assistants based on course material
- Legal or policy document summarization
- Personal knowledge bases from books or notes

---

## 🔒 Security Notes

- Keep your `.env` file secret and avoid committing it to version control.
- Be aware of OpenAI API rate limits and usage charges.
- Don’t expose your API to the public without proper access control.

---
