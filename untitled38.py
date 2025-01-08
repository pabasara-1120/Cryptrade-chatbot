# -*- coding: utf-8 -*-
"""Untitled38.ipynb"""

# %pip install chromadb --quiet
# %pip install sentence_transformers --quiet
# %pip install pypdf --quiet
# %pip install langchain --quiet
# %pip install tqdm --quiet
# %pip install google-generativeai --quiet

import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

from pypdf import PdfReader

from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import Client, PersistentClient
from chromadb.utils import embedding_functions

import textwrap
from IPython.display import display, Markdown

import os
import google.generativeai as genai
from dotenv import load_dotenv

# API key configuration
load_dotenv()

# Access the GOOGLE_API_KEY from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ChromaDB Configuration
chromaDB_path = './my_chroma_db'
collection_name = "Papers"
sentence_transformer_model = "distiluse-base-multilingual-cased-v1"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=sentence_transformer_model)


import os
from pathlib import Path

local_docs_path = Path('./local_docs')

# Collect all PDF and text file paths
pdf_file_paths = list(local_docs_path.glob('*.pdf'))
text_file_paths = list(local_docs_path.glob('*.txt'))


# Helper functions
def create_chroma_client(collection_name, embedding_function, chromaDB_path):
    if chromaDB_path is not None:
        chroma_client = PersistentClient(
            path=chromaDB_path,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE)
    else:
        chroma_client = Client()

    chroma_collection = chroma_client.get_or_create_collection(
        collection_name,
        embedding_function=embedding_function)

    return chroma_client, chroma_collection

# Initialize Chroma client and collection
chroma_client, chroma_collection = create_chroma_client(
    collection_name, embedding_function, chromaDB_path)

# Verify collection properties


# Helper function to convert text to markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

import requests












def convert_PDF_Text(pdf_path):
  reader = PdfReader(pdf_path)
  pdf_texts = [p.extract_text().strip() for p in reader.pages]
  pdf_texts = [text for text in pdf_texts if text]
  print("Document: ",pdf_path," chunk size: ", len(pdf_texts))
  return pdf_texts

def convert_Page_ChunkinChar(pdf_texts, chunk_size = 1500, chunk_overlap=0 ):
  character_splitter = RecursiveCharacterTextSplitter(
      separators=["\n\n", "\n", ". ", " ", ""],
      chunk_size=1500,
      chunk_overlap=0
)
  character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
  print(f"\nTotal number of chunks (document splited by max char = 1500): \
        {len(character_split_texts)}")
  return character_split_texts

def convert_Chunk_Token(text_chunksinChar,sentence_transformer_model, chunk_overlap=0,tokens_per_chunk=128 ):
  token_splitter = SentenceTransformersTokenTextSplitter(
      chunk_overlap=0,
      model_name=sentence_transformer_model,
      tokens_per_chunk=128)

  text_chunksinTokens = []
  for text in text_chunksinChar:
      text_chunksinTokens += token_splitter.split_text(text)
  print(f"\nTotal number of chunks (document splited by 128 tokens per chunk):\
       {len(text_chunksinTokens)}")
  return text_chunksinTokens

def add_meta_data(text_chunksinTokens, title, category, initial_id):
  ids = [str(i+initial_id) for i in range(len(text_chunksinTokens))]
  metadata = {
      'document': title,
      'category': category
  }
  metadatas = [ metadata for i in range(len(text_chunksinTokens))]
  return ids, metadatas


def add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection):
  print("Before inserting, the size of the collection: ", chroma_collection.count())
  chroma_collection.add(ids=ids, metadatas= metadatas, documents=text_chunksinTokens)
  print("After inserting, the size of the collection: ", chroma_collection.count())
  return chroma_collection

def load_agile_data_to_ChromaDB(collection_name, sentence_transformer_model, chromaDB_path):
    category = "Agile Documentation"
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=sentence_transformer_model)
    chroma_client, chroma_collection = create_chroma_client(collection_name, embedding_function, chromaDB_path)
    current_id = chroma_collection.count()

    # Fetch GitHub Data
    
    github_data = github_issues + github_milestones

    # Fetch Bugzilla Data
    
    # Add Bugzilla fetching here

    # Fetch Apache JIRA Data
    # Add JIRA fetching here

    # Combine all Agile data
    agile_data = github_data  # Add Bugzilla and JIRA data to this list as needed

    # Process Agile data
    for text in agile_data:
        text_chunksinChar = convert_Page_ChunkinChar([text])
        text_chunksinTokens = convert_Chunk_Token(text_chunksinChar, sentence_transformer_model)
        ids, metadatas = add_meta_data(text_chunksinTokens, title="Agile Data", category=category, initial_id=current_id)
        current_id += len(text_chunksinTokens)
        chroma_collection = add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection)

    # Process Local PDFs
    for pdf_path in pdf_file_paths:
        pdf_texts = convert_PDF_Text(pdf_path)  # Get text from each PDF
        for text in pdf_texts:
            text_chunksinChar = convert_Page_ChunkinChar([text])
            text_chunksinTokens = convert_Chunk_Token(text_chunksinChar, sentence_transformer_model)
            ids, metadatas = add_meta_data(text_chunksinTokens, title=pdf_path.stem, category="Local PDF", initial_id=current_id)
            current_id += len(text_chunksinTokens)
            chroma_collection = add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection)

    # Process Local Text Files
    for text_file_path in text_file_paths:
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
            text_chunksinChar = convert_Page_ChunkinChar([text_content])
            text_chunksinTokens = convert_Chunk_Token(text_chunksinChar, sentence_transformer_model)
            ids, metadatas = add_meta_data(text_chunksinTokens, title=text_file_path.stem, category="Local Text File", initial_id=current_id)
            current_id += len(text_chunksinTokens)
            chroma_collection = add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection)

    return chroma_client, chroma_collection




def process_and_add_to_chroma(text, title, current_id, chroma_collection, sentence_transformer_model):
    text_chunksinChar = convert_Page_ChunkinChar([text])
    text_chunksinTokens = convert_Chunk_Token(text_chunksinChar, sentence_transformer_model)
    ids, metadatas = add_meta_data(text_chunksinTokens, title, "Agile Documentation", current_id)
    add_document_to_collection(ids, metadatas, text_chunksinTokens, chroma_collection)

# Functions to retrieve and show documents from ChromaDB
def retrieveDocs(chroma_collection, query, n_results=5, return_only_docs=False):
    results = chroma_collection.query(query_texts=[query],
                                      include=["documents", "metadatas", 'distances'],
                                      n_results=n_results)

    if return_only_docs:
        return results['documents'][0]
    else:
        return results

def show_results(results, return_only_docs=False):
    if return_only_docs:
        retrieved_documents = results
        if len(retrieved_documents) == 0:
            print("No results found.")
            return
        for i, doc in enumerate(retrieved_documents):
            print(f"Document {i+1}:")
            print("\tDocument Text: ")
            display(to_markdown(doc))
    else:
        retrieved_documents = results['documents'][0]
        retrieved_documents_metadata = results['metadatas'][0]
        retrieved_documents_distances = results['distances'][0]
        print("------- Retrieved Documents -------\n")
        for i, doc in enumerate(retrieved_documents):
            print(f"Document {i+1}:")
            print("\tDocument Text: ")
            display(to_markdown(doc))
            print(f"\tDocument Source: {retrieved_documents_metadata[i]['document']}")
            print(f"\tDocument Source Type: {retrieved_documents_metadata[i]['category']}")
            print(f"\tDocument Distance: {retrieved_documents_distances[i]}")


results = retrieveDocs(chroma_collection, query="Explain Agile sprint planning", n_results=5)
#print("Results:", results)
#show_results(results)


def build_chatBot(system_instruction):
  model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=system_instruction)
  chat = model.start_chat(history=[])
  return chat

def generate_LLM_answer(prompt, context, chat):
  response = chat.send_message( prompt + context)
  return response.text

system_prompt = """
"""

RAG_LLM = build_chatBot(system_prompt)

def generateAnswer(RAG_LLM, chroma_collection,query,n_results=5, only_response=True):
    retrieved_documents= retrieveDocs(chroma_collection, query, 10, return_only_docs=True)
    prompt = "QUESTION: "+ query
    context = "\n EXCERPTS: "+ "\n".join(retrieved_documents)
    if not only_response:
      print("------- retreived documents -------\n")
      for i, doc in enumerate(retrieved_documents):
        print(f"Document {i+1}:")
        print(f"\tDocument Text: {doc}")
      print("------- RAG answer -------\n")
    output = generate_LLM_answer(prompt, context, RAG_LLM)

    display(to_markdown(output))
    print('\n')
    return output

query = """"""

reply=generateAnswer(RAG_LLM, chroma_collection, query,10, only_response=False)

print(reply)


def initialize_rag():
    """Initialize RAG components for FastAPI"""
    chroma_client, chroma_collection = create_chroma_client(
        collection_name,
        embedding_function,
        chromaDB_path
    )

    system_prompt = """
"""

    model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=system_prompt)
    chat = model.start_chat(history=[])

    return chroma_collection, chat


def process_chat_query(query: str, chroma_collection, rag_llm):
    """Process a single chat query"""
    retrieved_documents = retrieveDocs(
        chroma_collection,
        query,
        n_results=5,
        return_only_docs=True
    )

    prompt = "QUESTION: " + query
    context = "\nEXCERPTS: " + "\n".join(retrieved_documents)

    response = generate_LLM_answer(prompt, context, rag_llm)
    return response




# Building a chatbot with Google Generative AI
