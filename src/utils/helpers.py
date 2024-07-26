from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
import os, sys, getpass, dotenv, pathlib
from sqlalchemy.orm import Session
from langchain_community.llms.ollama import Ollama
from tiktoken.load import load_tiktoken_bpe
import tiktoken
from langchain_community.vectorstores.redis import Redis


print(f"👉 getting ENVs from {os.getcwd()}/.env")
# TODO to add all kind of different names, e.g: .env.local, .env.dev, etc.
# TODO to check if cwd is outside of src and so on
dotenv.load_dotenv(f"{os.getcwd()}/.env")
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}

    ---
    Answer the question based on the above context: {context}
    """
print("👉 DB on", DATABASE_URL)


def load_documents(path):
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=80,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def split_document(documents: list[Document]):
    # num_reserved_special_tokens = 256

    # pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    # special_tokens = [
    #     "<|begin_of_text|>",
    #     "<|end_of_text|>",
    #     "<|reserved_special_token_0|>",
    #     "<|reserved_special_token_1|>",
    #     "<|reserved_special_token_2|>",
    #     "<|reserved_special_token_3|>",
    #     "<|start_header_id|>",
    #     "<|end_header_id|>",
    #     "<|reserved_special_token_4|>",
    #     "<|eot_id|>",  # end of turn
    # ] + [
    #     f"<|reserved_special_token_{i}|>"
    #     for i in range(5, num_reserved_special_tokens - 5)
    # ]
    # mergeable_ranks = load_tiktoken_bpe(model_path)
    # num_base_tokens = len(mergeable_ranks)
    # special_tokens = {
    #     token: num_base_tokens + i for i, token in enumerate(special_tokens)
    # }

    # tt = TokenTextSplitter()

    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
    # return tt.split_documents(documents)


def get_embedding_function():

    # Embedding function. Key For the VectorDatabase
    embeddings = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="llama3",
    )
    return embeddings


def add_to_vector_database(path: str):

    docs = load_documents(path)

    chunks = split_documents(docs)

    embeddings_model = HuggingFaceEmbeddings()
    embeddings = embeddings_model.embed
    # db = PGEmbedding(
    #     embedding_function=get_embedding_function(),
    #     collection_name=COLLECTION_NAME,
    #     connection_string=DATABASE_URL,
    # )

    # Calculate Page IDs.

    # # if len(chunks):
    # print(f"👉 Adding {(chunks)} chunks of documents: ")

    # res = db.add_texts(chunks)
    # # db.persist()

    # # else:
    # #     print("✅ No new documents to add")

    # print("✅ Done Adding documents")
    # print("Added", res[-1])


def add_to_redis_VD(chunks: list[Document]):
    rds = Redis.from_texts(
        texts=chunks,
        embedding=get_embedding_function(),
        metadatas=[chunk.metadata for chunk in chunks],
        redis_url=REDIS_URL,
        index_name=COLLECTION_NAME,
    )


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def query_rag(query_text: str, db):

    # db = PGEmbedding(
    #     embedding_function=get_embedding_function(),
    #     collection_name=COLLECTION_NAME,
    #     connection_string=DATABASE_URL,
    # )
    try:
        sess = Session(db.connect())

        results = db.similarity_search_with_score(query=query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(prompt)
        model = Ollama(model="llama3")
        response_text = model.invoke(prompt)
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        return response_text
    except:
        print("Error, db is not a VectorStore ")
