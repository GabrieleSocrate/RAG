"""
A RAG system is mainly formed by 5 parts:
1) Document Loaders: load very different types of data like PDFs, text files....
2) Text Splitting: text splitters break large documents into smaller chunks.
3) Indexing: after splitting you need to organize the data. Using indexing you 
turn text chunks into vectors.
4) Retrieval Models: they have to search through all the indexed data and find 
what you need.
    a) Vector Stores: databases designed to handle vector representations of the 
    text chinks. They use vector similarity search between the query and the stored 
    vectors.
    b) Retrievers: they actually do the searching, they take the user's query, convert it
    into a vector and then search in the vector store to find most relevant data
5) Generative Models: once you retrieved the relevant data, the generative models
produce the final response.
"""

"""The API key is inside the .env file to get access to it we use dotenv package to load enviroment variables"""
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY nt found")

# -----------------------------------------
# 1) 
# DOCUMENT LOADERS
# -----------------------------------------
from langchain_community.document_loaders import PyPDFLoader
all_files = ["Paper/ASurvey on RAGMeetingLLMs Towards Retrieval-Augmented Large Language Models.pdf",
            "Paper/Attention Is All You Need.pdf",
            "Paper/Dense Passage Retrieval for Open-Domain Question Answering.pdf",
            "Paper/REALM Retrieval-Augmented Language Model Pre-Training.pdf",
            "Paper/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf",
            "Paper/Retrieval-Augmented Generation for Large Language Models A Survey.pdf",
            "Paper/Seven Failure Points When Engineering a Retrieval Augmented Generation System.pdf"]
documents = []
for file in all_files:
    loader = PyPDFLoader(file) 
    document = loader.load()
    documents.extend(document) 
    # here we use extend and not append since with append you the element document to the list documents, 
    # but document is a lis of pages, so you obtain a list of lists. Instead using extend you add the single 
    # elements of the list document to the list documents so you will end up with just a list

# ----------------------------------------
# 2)
# CHUNKING
# ----------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
document_chunks = splitter.split_documents(documents)
# I'm splitting the documents into chunks of 500 characters each. To avoid cutting a concept in half I set an 
# overlap between chunks of 50 characters.

# -----------------------------------------
# 3) 
# EMBEDDINGS
# -----------------------------------------
from langchain_openai.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
# For vector embeddings I use model from OpenAI. So we use it to transform chunks (text) into vector of numbers.

# -----------------------------------------
# 4) 
# VECTOR STORES
# -----------------------------------------
from langchain_community.vectorstores import FAISS
if os.path.exists("faiss_index"): # if we have already done the embeddings
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings, # we have to specify the embedding model we used
        allow_dangerous_deserialization=True
    )
else: # THIS IS FOR THE FIRST RUN (we create for the first time the embeddings)
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local("faiss_index") # We save the embeddings locally otherwise I should pay for embeddings every time I run this file, since I'd do the embeddings every time
# This is the vector store (FAISS) where we store all the vectors we have obtained from embedding

# -----------------------------------------
# 5)
# RETRIEVAL
# -----------------------------------------
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 10}
)
# So at first we convert the vector store into a retriever, and then we ask for the 10 most relevant chunks

# -----------------------------------------
# 6) 
# QUERYING
# -----------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0) # with temperture = 0 the answer is less creative and more deterministic

async def rag_answer(question):
    """
    The RAG function is asynchronous because it performs operations that involve waiting time (retrieve data,
    build context, generate a response). While a single user request is waiting for these operations to complete,
    making the function asynchronous allows the server to handle other user requests at the same time, instead 
    of being blocked by one request. The execution of the RAG function itself continues normally for that user, 
    but it does not monopolize the server while waiting. This improves concurrency and allows multiple users to 
    receive responses in parallel.
    """
    docs = retriever.invoke(question) # we get the 10 most relevant chunks
    context = "\n\n---\n\n".join(doc.page_content for doc in docs) # join all the chunks creating one long string
    messages = [
        SystemMessage(content = "Answer using only the provided context. If not present, say you don't know."),
        HumanMessage(content = f"Question:\n{question}\n\nContext:\n{context}")
    ]

    answer = llm.invoke(messages).content
    return answer

"""
The system message defines the global behavior of the model. It sets rules, constraints, and the style of the answer 
(for example, “use only the provided context” or “do not invent information”). The human (user) message represents 
the user’s input and is the channel used to provide the question and, in RAG systems, the supporting information 
retrieved from documents. The assistant message is the response generated by the model. By explicitly separating these roles, 
the model can distinguish between instructions, requests, and responses, improving control over its behavior and reducing hallucinations.
"""