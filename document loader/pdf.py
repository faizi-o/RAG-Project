

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  

chunk = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

data = PyPDFLoader("sample.pdf")

docs = data.load()

split = chunk.split_documents(docs)
# print (docs[0].page_content)
print (split)

