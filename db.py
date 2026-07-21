from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("sample.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

embeddings = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embeddings,
    persist_directory = "chroma_db")
