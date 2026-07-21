
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Page title
st.title("RAG Document Assistant")
st.write("Ask questions from your document using Retrieval-Augmented Generation.")

# Initialize RAG system
@st.cache_resource
def load_rag_system():

    embeddings = MistralAIEmbeddings()

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    llm = ChatMistralAI(
        model="mistral-small-2603"
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
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

    return retriever, llm, prompt


# Load RAG components
retriever, llm, prompt = load_rag_system()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
query = st.chat_input("Ask a question about the document...")


if query:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Searching the document..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            result = llm.invoke(final_prompt)

            answer = result.content

            st.markdown(answer)


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# import streamlit as st
# from dotenv import load_dotenv
# from tempfile import NamedTemporaryFile

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate


# load_dotenv()


# st.set_page_config(
#     page_title="PDF RAG Assistant",
#     page_icon="📄"
# )


# st.title("📄 PDF RAG Assistant")
# st.write("Upload a PDF and ask questions about its content.")


# # Create embeddings and LLM only once
# @st.cache_resource
# def load_models():

#     embeddings = MistralAIEmbeddings()

#     llm = ChatMistralAI(
#         model="mistral-small-2603"
#     )

#     return embeddings, llm


# embeddings, llm = load_models()


# # PDF upload
# uploaded_file = st.file_uploader(
#     "Upload a PDF",
#     type=["pdf"]
# )


# # Process uploaded PDF
# if uploaded_file is not None:

#     # Avoid processing the same PDF again
#     if (
#         "file_name" not in st.session_state
#         or st.session_state.file_name != uploaded_file.name
#     ):

#         with st.spinner("Processing PDF..."):

#             # Save uploaded file temporarily
#             with NamedTemporaryFile(
#                 delete=False,
#                 suffix=".pdf"
#             ) as temp_file:

#                 temp_file.write(
#                     uploaded_file.getbuffer()
#                 )

#                 pdf_path = temp_file.name


#             # Load PDF
#             loader = PyPDFLoader(pdf_path)

#             docs = loader.load()


#             # Split text
#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200
#             )

#             chunks = splitter.split_documents(docs)


#             # Create vector database in memory
#             vector_store = Chroma.from_documents(
#                 documents=chunks,
#                 embedding=embeddings
#             )


#             # Save vector store in session
#             st.session_state.vector_store = vector_store

#             st.session_state.file_name = uploaded_file.name


#         st.success(
#             f"{uploaded_file.name} processed successfully!"
#         )


# # Ask questions
# query = st.chat_input(
#     "Ask a question about the PDF..."
# )


# if query:

#     if "vector_store" not in st.session_state:

#         st.warning(
#             "Please upload a PDF first."
#         )

#         st.stop()


#     # Retrieve relevant documents
#     retriever = st.session_state.vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 4,
#             "fetch_k": 10,
#             "lambda_mult": 0.5
#         }
#     )


#     docs = retriever.invoke(query)


#     # Create context
#     context = "\n\n".join(
#         [doc.page_content for doc in docs]
#     )


#     # Prompt
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are a helpful AI assistant.

# Use ONLY the provided context to answer the question.

# If the answer is not present in the context, say:

# I could not find the answer in the document.

# Do not use outside knowledge.
# """
#             ),
#             (
#                 "human",
#                 """Context:
# {context}

# Question:
# {question}
# """
#             )
#         ]
#     )


#     final_prompt = prompt.invoke(
#         {
#             "context": context,
#             "question": query
#         }
#     )


#     # Generate answer
#     result = llm.invoke(final_prompt)


#     st.write(result.content)