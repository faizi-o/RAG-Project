from dotenv import load_dotenv
load_dotenv()   
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

embeddings = MistralAIEmbeddings()

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
    )


llm = ChatMistralAI(model="mistral-small-2603")

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4 ,
                   "fetch_k": 10,
                   "lambda_mult": 0.5}
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print ("RAG system is ready. Type '0' to quit.")

while True:
    query = input("Enter your question: ")
    if query == "0":
        break

    docs = retriever.invoke(query)


    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": query})

    result = llm.invoke(final_prompt)

    print(result.content)

