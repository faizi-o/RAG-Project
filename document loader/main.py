from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("test.txt")
splitter = CharacterTextSplitter(
    separator="",
    chunk_size=30,
    chunk_overlap=1)

docs = data.load()

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print ()
    