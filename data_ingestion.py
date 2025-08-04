from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# Retrieving the complete response as a list from WebBaseLoader
docs = [WebBaseLoader(url).load() for url in urls]

# Flattening the structure to get the list of documents
doc_list = [item for sublist in docs for item in sublist]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    add_start_index=True
)

docs_split = text_splitter.split_documents(doc_list)

# Should be executed only once
vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embedding_model,
    collection_name="rag-docs",
    persist_directory="./.chroma",
)

retriever = Chroma(
    collection_name="rag-docs",
    persist_directory="./.chroma",
    embedding_function=embedding_model,
).as_retriever()
