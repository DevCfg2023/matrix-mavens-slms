
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant


QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def refresh_data(collection_name: str):
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # Load without re-uploading

    vectorstore =  Qdrant(
        collection_name=collection_name,
        embedding_function=embedder,
        url=QDRANT_URL,
    )

    return vectorstore


def load_vector_store(
        pdf_path: str,
        index_name: str,
        vector_store_type: str = "Qdrant",
):
    """Load vector store from PDF, supports Qdrant and Pinecone"""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if vector_store_type == "Qdrant":
        store = Qdrant.from_documents(
            documents=chunks,
            embedding=embedder,
            url=QDRANT_URL,
            prefer_grpc=False,
            collection_name=index_name
        )
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")

    return store
