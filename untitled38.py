# -*- coding: utf-8 -*-
"""Crypto Trading Chatbot"""

# Required installations:
# %pip install chromadb --quiet
# %pip install sentence_transformers --quiet
# %pip install pypdf --quiet
# %pip install langchain --quiet
# %pip install tqdm --quiet
# %pip install google-generativeai --quiet
# %pip install python-dotenv --quiet

import os
from pathlib import Path
import textwrap
import google.generativeai as genai
from dotenv import load_dotenv
from IPython.display import display, Markdown
from pypdf import PdfReader
from chromadb import Client, PersistentClient
from chromadb.utils import embedding_functions
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import redis
import requests
import json
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Configuration
chromaDB_path = './my_chroma_db'
collection_name = "CryptoTrading"
sentence_transformer_model = "distiluse-base-multilingual-cased-v1"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=sentence_transformer_model
)

# Local document path
local_docs_path = Path('./local_docs')
pdf_file_paths = list(local_docs_path.glob('*.pdf'))
text_file_paths = list(local_docs_path.glob('*.txt'))

# Helper Functions
def create_chroma_client(collection_name, embedding_function, chromaDB_path):
    chroma_client = PersistentClient(
        path=chromaDB_path,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )
    chroma_collection = chroma_client.get_or_create_collection(
        collection_name,
        embedding_function=embedding_function
    )
    return chroma_client, chroma_collection

def convert_PDF_Text(pdf_path):
    reader = PdfReader(pdf_path)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    pdf_texts = [text for text in pdf_texts if text]
    print("Document:", pdf_path, "Chunk size:", len(pdf_texts))
    return pdf_texts

def convert_Page_ChunkinChar(pdf_texts, chunk_size=1500, chunk_overlap=0):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return character_splitter.split_text('\n\n'.join(pdf_texts))

def convert_Chunk_Token(text_chunksinChar, sentence_transformer_model, chunk_overlap=0, tokens_per_chunk=128):
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        model_name=sentence_transformer_model,
        tokens_per_chunk=tokens_per_chunk
    )
    text_chunksinTokens = []
    for text in text_chunksinChar:
        text_chunksinTokens += token_splitter.split_text(text)
    return text_chunksinTokens

def add_meta_data(text_chunksinTokens, title, category, initial_id):
    ids = [str(i + initial_id) for i in range(len(text_chunksinTokens))]
    metadata = {'document': title, 'category': category}
    metadatas = [metadata for _ in text_chunksinTokens]
    return ids, metadatas

def add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection):
    print("Before inserting, size of collection:", chroma_collection.count())
    chroma_collection.add(ids=ids, metadatas=metadatas, documents=text_chunksinTokens)
    print("After inserting, size of collection:", chroma_collection.count())

def load_crypto_data_to_ChromaDB(collection_name, sentence_transformer_model, chromaDB_path):
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=sentence_transformer_model
    )
    chroma_client, chroma_collection = create_chroma_client(
        collection_name, embedding_function, chromaDB_path
    )
    current_id = chroma_collection.count()

    for pdf_path in pdf_file_paths:
        pdf_texts = convert_PDF_Text(pdf_path)
        for text in pdf_texts:
            text_chunksinChar = convert_Page_ChunkinChar([text])
            text_chunksinTokens = convert_Chunk_Token(text_chunksinChar, sentence_transformer_model)
            ids, metadatas = add_meta_data(text_chunksinTokens, title=pdf_path.stem, category="Crypto PDF", initial_id=current_id)
            current_id += len(text_chunksinTokens)
            add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection)

    for text_file_path in text_file_paths:
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
            text_chunksinChar = convert_Page_ChunkinChar([text_content])
            text_chunksinTokens = convert_Chunk_Token(text_chunksinChar, sentence_transformer_model)
            ids, metadatas = add_meta_data(text_chunksinTokens, title=text_file_path.stem, category="Crypto Text File", initial_id=current_id)
            current_id += len(text_chunksinTokens)
            add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection)

    return chroma_client, chroma_collection

def retrieveDocs(chroma_collection, query, n_results=5, return_only_docs=False):
    results = chroma_collection.query(
        query_texts=[query],
        include=["documents", "metadatas", 'distances'],
        n_results=n_results
    )
    if return_only_docs:
        return results['documents'][0]
    return results

def show_results(results):
    retrieved_documents = results['documents'][0]
    retrieved_documents_metadata = results['metadatas'][0]
    print("------- Retrieved Documents -------\n")
    for i, doc in enumerate(retrieved_documents):
        print(f"Document {i + 1}:")
        print("\tDocument Text:")
        display(Markdown(textwrap.indent(doc, '> ', predicate=lambda _: True)))
        print(f"\tDocument Source: {retrieved_documents_metadata[i]['document']}")
        print(f"\tCategory: {retrieved_documents_metadata[i]['category']}\n")

def build_chatBot(system_instruction):
    model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=system_instruction)
    chat = model.start_chat(history=[])
    return chat

def generate_LLM_answer(prompt, context, chat):
    response = chat.send_message(prompt + "\n" + context)
    return response.text

def generateAnswer(RAG_LLM, chroma_collection, query, n_results=5, jwt_token=None):
    # Retrieve documents from ChromaDB
    retrieved_documents = retrieveDocs(chroma_collection, query, n_results, return_only_docs=True)
    
    # Fetch real-time data
    try:
        api_url = "http://localhost:8080/coins/top50"
        real_time_data = fetch_and_cache_real_time_data(api_url, cache_key="crypto_real_time")
        print("Real-time data fetched")
    except Exception as e:
        real_time_data = {"error": str(e)}
    
    # Fetch user-specific data
    user_watchlist_data = {}
    if jwt_token:
        user_watchlist_data = fetch_user_watchlist(jwt_token)
    
    # Combine data into the prompt
    prompt = f"QUESTION: {query}"
    context = "\nEXCERPTS:\n" + "\n".join(retrieved_documents)
    context += f"\nREAL-TIME DATA:\n{json.dumps(real_time_data, indent=2)}"
    context += f"\nUSER WATCHLIST DATA:\n{json.dumps(user_watchlist_data, indent=2)}"
    
    # Generate answer using LLM
    return generate_LLM_answer(prompt, context, RAG_LLM)

def fetch_user_watchlist(jwt_token):
    """Fetch user-specific data, using Redis as a cache."""
    cache_key = f"user_watchlist:{jwt_token}"
    
    # Check Redis cache
    cached_data = redis_client.get(cache_key)
    if cached_data:
        print("User watchlist data fetched from cache")
        return json.loads(cached_data)
    
    # If not in cache, fetch from Spring API
    user_api_url = "http://localhost:8080/api/watchlist/user"
    headers = {"Authorization": f"Bearer {jwt_token}"}
    
    try:
        response = requests.get(user_api_url, headers=headers)
        response.raise_for_status()
        user_watchlist_data = response.json()
        
        # Store in Redis cache with a 10-minute expiration
        redis_client.setex(cache_key, 600, json.dumps(user_watchlist_data))
        print("User watchlist data fetched from API and cached")
        return user_watchlist_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user watchlist: {e}")
        return {"error": str(e)}


def initialize_rag():
    """Initialize RAG components for FastAPI"""
    try:
        # Create or load the ChromaDB client and collection
        chroma_client = PersistentClient(
            path=chromaDB_path,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        chroma_collection = chroma_client.get_or_create_collection(
            collection_name,
            embedding_function=embedding_function
        )
        print("ChromaDB collection successfully created or loaded.")

        # Optionally load documents into the ChromaDB collection if it's empty
        if chroma_collection.count() == 0:
            print("ChromaDB collection is empty. Loading data into ChromaDB...")
            load_crypto_data_to_ChromaDB(
                collection_name, 
                sentence_transformer_model, 
                chromaDB_path
            )
            print("Data successfully loaded into ChromaDB.")
        else:
            print(f"ChromaDB collection already contains {chroma_collection.count()} documents.")

        # Define the system prompt for the generative AI chatbot
        system_prompt = """
        You are a knowledgeable assistant specializing in cryptocurrency trading. 
        Use the provided context and retrieved documents to answer user queries.
        Be as specific and informative as possible, citing your sources where relevant.
        """

        # Initialize the generative AI chatbot with the system prompt
        model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=system_prompt)
        chat = model.start_chat(history=[])

        # Return the initialized components
        return chroma_collection, chat

    except Exception as e:
        print(f"Error initializing RAG components: {e}")
        raise RuntimeError(f"Failed to initialize RAG components: {e}")








system_prompt = "You are a knowledgeable assistant specializing in cryptocurrency trading."
RAG_LLM = build_chatBot(system_prompt)

redis_client = redis.StrictRedis(host='127.0.0.1', port=6379, decode_responses=True)


def fetch_and_cache_real_time_data(api_url, cache_key, ttl=300):
    cached_data = redis_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    
    response = requests.get(api_url)
    if response.status_code == 200:
        real_time_data = response.json()
        redis_client.set(cache_key, json.dumps(real_time_data))
        redis_client.expire(cache_key, ttl)
        return real_time_data
    else:
        raise Exception(f"Failed to fetch real-time data: {response.status_code}")


