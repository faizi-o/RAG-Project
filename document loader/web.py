from langchain_community.document_loaders import WebBaseLoader

data = WebBaseLoader("https://hico.pk/")

docs = data.load()

print (len(docs))