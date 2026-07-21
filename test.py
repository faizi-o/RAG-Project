import langchain
import sys
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import  ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("sample.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

template = chat_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are an AI assistant that summarizes the text"),
        ("human", "{data}")

    ]
)


model = ChatMistralAI(model ="mistral-small-2603")

prompt = template.format_prompt(data=docs)

result = model.invoke(prompt)

print (result.content)
