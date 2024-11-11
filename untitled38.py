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

# API key configuration
GOOGLE_API_KEY = 'AIzaSyB-QRYybxvfsioAEddtygVDTa6dmn2BE9I'
genai.configure(api_key=GOOGLE_API_KEY)

# ChromaDB Configuration
chromaDB_path = './my_chroma_db'
collection_name = "Papers"
sentence_transformer_model = "distiluse-base-multilingual-cased-v1"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=sentence_transformer_model)


import os
from pathlib import Path

# Path to the folder containing local PDFs and text files
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

def fetch_bugzilla_issues(product='Firefox', status='NEW', max_results=15):
    """
    Fetch recent issues from Mozilla's Bugzilla for a specified product.
    """
    url = 'https://bugzilla.mozilla.org/rest/bug'
    params = {'product': product, 'status': status, 'limit': max_results}
    response = requests.get(url, params=params)
    response.raise_for_status()  # This will raise an error if the request fails
    bugs = response.json().get('bugs', [])

    bug_texts = []
    for bug in bugs:
        text = f"Bug #{bug['id']}: {bug['summary']} (Assigned to: {bug['assigned_to']})"
        bug_texts.append(text)

    return bug_texts

# Test the function with a smaller limit for simplicity




def fetch_jira_issues(jql, max_results=5):
    """
    Fetch issues from Apache JIRA using the specified JQL query.
    """
    url = 'https://issues.apache.org/jira/rest/api/2/search'
    params = {'jql': jql, 'maxResults': max_results}
    response = requests.get(url, params=params)
    response.raise_for_status()  # This will raise an error if the request fails
    issues = response.json().get('issues', [])

    jira_texts = []
    for issue in issues:
        text = f"Issue {issue['key']}: {issue['fields']['summary']} (Status: {issue['fields']['status']['name']})"
        jira_texts.append(text)

    return jira_texts

# Test the function with a sample JQL query



def fetch_github_issues(repo_owner, repo_name):
    issues_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/issues'
    response = requests.get(issues_url)
    issues = response.json() if response.status_code == 200 else []
    issue_texts = [f"Issue {issue['number']}: {issue['title']} - {issue['body']}" for issue in issues if 'body' in issue]
    return issue_texts

def fetch_github_milestones(repo_owner, repo_name):
    milestones_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/milestones'
    response = requests.get(milestones_url)
    milestones = response.json() if response.status_code == 200 else []
    milestone_texts = [f"Milestone {milestone['title']}: Due on {milestone['due_on']}" for milestone in milestones]
    return milestone_texts


agile_manifesto_text = [
    "Our highest priority is to satisfy the customer through early and continuous delivery of valuable software.",
"Welcome changing requirements, even late in development. Agile processes harness change for the customer's competitive advantage.",

"Deliver working software frequently, from a couple of weeks to a couple of months, with a preference to the shorter timescale.",

"Business people and developers must work together daily throughout the project.",

"Build projects around motivated individuals.Give them the environment and support they need, and trust them to get the job done.",

"The most efficient and effective method of conveying information to and within a development team is face-to-face conversation.",

"Working software is the primary measure of progress.",

"Agile processes promote sustainable development. The sponsors, developers, and users should be able to maintain a constant pace indefinitely.",

"Continuous attention to technical excellence and good design enhances agility.",

"Simplicity--the art of maximizing the amount of work not done--is essential.",

"The best architectures, requirements, and designs emerge from self-organizing teams.",

"At regular intervals, the team reflects on how to become more effective, then tunes and adjusts its behavior accordingly.",
]
def convert_PDF_Text(pdf_path):
  reader = PdfReader(pdf_path)
  pdf_texts = [p.extract_text().strip() for p in reader.pages]
  # Filter the empty strings
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
def fetch_bugzilla_bugs(product_name):
    bugzilla_url = 'https://bugzilla.mozilla.org/rest/bug'
    params = {'product': product_name, 'limit': 10}  # Adjust limit as needed
    response = requests.get(bugzilla_url, params=params)
    bugs = response.json().get('bugs', []) if response.status_code == 200 else []
    bug_texts = [f"Bug #{bug['id']}: {bug['summary']} (Assigned to: {bug['assigned_to']})" for bug in bugs]
    return bug_texts

def fetch_apache_jira_issues(project_key):
    jira_url = 'https://issues.apache.org/jira/rest/api/2/search'
    params = {'jql': f'project={project_key} AND status=Open', 'maxResults': 10}
    response = requests.get(jira_url, params=params)
    issues = response.json().get('issues', []) if response.status_code == 200 else []
    jira_texts = [f"Issue {issue['key']}: {issue['fields']['summary']} (Status: {issue['fields']['status']['name']})" for issue in issues]
    return jira_texts

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
    github_issues = fetch_github_issues('apache', 'kafka')  # Example repo
    github_milestones = fetch_github_milestones('apache', 'kafka')
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
You're an AI software process guide specialized in Agile methodologies.
Your purpose is to assist users with detailed, practical advice on Agile process activities and best practices,
helping them navigate and implement activities such as sprint planning, backlog refinement,
daily standups, sprint reviews, retrospectives, release planning, and more. You have in-depth knowledge of Agile frameworks and can guide users through 10+ key process activities in software development.In other words, the agile process model should be evaluated against a minimum of 10 different process activities
Provide clear, step-by-step instructions, answer questions, and offer relevant resources or examples. Ensure your responses are accurate, practical, and structured, helping users understand Agile practices better.
You can pull relevant context to enhance clarity and adapt explanations based on the user’s familiarity with Agile concepts, whether they are beginners or advanced practitioners.
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

query = """How can I improve our backlog refinement process to keep it organized and up-to-date?"""

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
You're an AI software process guide and assistant with a focus on Scum methodologies, supporting a broad range of Agile process activities. The process model we use here is Scrum.
Your purpose is to help users by providing practical, structured guidance across a minimum of 10 key Scrum process activities, including:

1. User stories, Persona Discovery and Creation 
2. Product Backlog, 
3. Sprint Planning, Backlog Refinement, Sprint Backlog, 
4.. Daily Standups   →  daily summaries of team progress
5.. Sprint Review  → Sprint reports, summaries 
6. Retrospective 
7. Release Planning (Predictive Analytics) (using feedback) 
8. Incident/Bug Management 
9. Continuous Integration/Deployment (CI/CD) 
10. Quality Assurance/Testing (e.g: Test Data Generation, identify edge cases, and even predict areas of the code that might fail) 
11. Stakeholder Communication
12. Product Documentation 


Provide clear, step-by-step instructions and ensure responses are tailored to these specific areas of Scrum practice. When responding to a question, indicate which of the 10 activities it aligns with and ensure that guidance is actionable and easy to understand. Include relevant resources, examples, and concise summaries to reinforce understanding.

*Note:* Maintain awareness of the user's level of experience and adapt explanations to suit both beginners and advanced practitioners. Ensure each answer is concise but fully addresses the specific Agile process activity. And refer to the context mainly and give them as references.
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
