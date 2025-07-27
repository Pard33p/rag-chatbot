from fastapi import FastAPI
from app.rag import create_vectorstore, query_rag
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
vectorstore = create_vectorstore("data/Assignment 2.pdf")

@app.get('/ask')
def ask(q: str):
    answer = query_rag(vectorstore, q)
    return {'answer': answer}