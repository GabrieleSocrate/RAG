# Retrieval-Augmented Generation (RAG) System

This repository contains a **Retrieval-Augmented Generation (RAG)** system built on top of academic papers.  
The goal of the project is to demonstrate how to design an **end-to-end RAG pipeline**, from document ingestion to answer generation, with a focus on **engineering choices** such as efficiency, modularity, and deployment readiness.

The project currently implements the **core RAG logic** in Python and will be extended with a **FastAPI layer** to expose the system as a production-ready API.

---

## **Project Overview**

The RAG system is composed of the following main components:

1. **Document Loading**  
   PDF papers are loaded using dedicated document loaders. Each page is converted into a structured document object that can be processed downstream.

2. **Text Chunking**  
   Large documents are split into smaller overlapping chunks to preserve semantic continuity while remaining compatible with embedding models.

3. **Embedding and Indexing**  
   Each chunk is converted into a vector representation using an embedding model.  
   The vectors are stored in a **FAISS vector store** to enable fast similarity search.

4. **Retrieval**  
   Given a user query, the system retrieves the **top-k most relevant chunks** from the vector store using vector similarity.

5. **Generation**  
   The retrieved context is injected into a prompt and passed to a language model, which generates a grounded answer based only on the retrieved information.

---

## **Cost-Aware Design**

To avoid recomputing embeddings at every run (and unnecessarily increasing API costs), the FAISS index is **persisted locally**:

- On the first run, embeddings are computed and saved to disk.
- On subsequent runs, the index is loaded directly from disk.

---

## **Running the API (FastAPI + Uvicorn)**

The RAG system is exposed through a FastAPI application, served using Uvicorn, an ASGI web server for Python.

To start the API locally, run:

uvicorn main:app --reload


Once the server is running, Uvicorn will print an address similar to:

http://127.0.0.1:8000

**Important Note**

Copy the printed address and manually add /docs:

http://127.0.0.1:8000/docs

