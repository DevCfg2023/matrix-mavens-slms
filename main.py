import os
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools import Tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import qdrant_client
import uuid

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import traceback
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from tempfile import NamedTemporaryFile
from vectorstore_loader import load_vector_store, refresh_data

app = FastAPI()
load_dotenv()

# ----------------------------------------
# 🔧 Configuration
# ----------------------------------------

VECTOR_DB_URL = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
VECTOR_STORE_TYPE = "Qdrant"

session_store = {}

# ----------------------------------------
# 🧠 Load or Refresh Store
# ----------------------------------------


def load_qdrant_store(collection_name: str):
    client = QdrantClient(url="http://localhost:6333")

    store = Qdrant(
        client=client,
        collection_name="finance_data",
        embeddings=embedder
    )
    return store


def create_or_refresh_store(collection_name: str, file_bytes: bytes):
    loader = PyMuPDFLoader(file_bytes)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)

    # Delete collection if it exists
    client = qdrant_client.QdrantClient(url=VECTOR_DB_URL)
    if collection_name in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection_name=collection_name)

    return Qdrant.from_documents(
        documents=chunks,
        embedding=embedder,
        url=VECTOR_DB_URL,
        prefer_grpc=True,
        collection_name=collection_name,
    )

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load vector store
        index_name = 'finance_data'  # str(uuid.uuid4())
        store = load_vector_store(
            pdf_path=tmp_path,
            index_name=index_name,
            vector_store_type=VECTOR_STORE_TYPE
        )

        # Create tools and agent
        pdf_tool = create_pdf_qa_tool(store)
        summary_tool = create_summary_tool(store)
        agent = build_agent(pdf_tool, summary_tool)

        # Create conversation memory for this session
        session_id = str(uuid.uuid4())
        session_store[session_id] = {
            "agent": agent,
            "memory": [],
        }
        return {"session_id": session_id}

    except Exception as e:
        traceback.print_exc()  # Logs the full traceback to console
        raise HTTPException(status_code=500, detail=str(e))




# ----------------------------------------
# 🛠️ Tools & LLM
# ----------------------------------------

def get_llm():
    if os.getenv("GROQ_API_KEY"):
        return ChatGroq(model="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))


def create_pdf_qa_tools(store, name="PDF_QA"):
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return Tool(name=name, func=chain.run, description="Answer questions from the PDF.")


def create_pdf_qa_tool(store, name="PDF_QA"):
    llm = get_llm()

    # Define a prompt explicitly
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}
        """
    )

    retriever = store.as_retriever(search_kwargs={"k": 3})

    # Build a pipeline manually without using deprecated LLMChain
    def retrieval_with_context(inputs):
        docs = retriever.get_relevant_documents(inputs["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context, "question": inputs["question"]}

    pipeline = (
        RunnableLambda(retrieval_with_context)
        | prompt
        | llm
    )

    def answer_question(question: str):
        return pipeline.invoke({"question": question})

    return Tool(
        name=name,
        func=answer_question,
        description="Answer questions from the PDF."
    )

def create_summary_tool(store, name="PDF_Summary"):
    prompt = PromptTemplate.from_template("Summarize the following content:\n\n{content}\n\nSummary:")
    chain = LLMChain(llm=get_llm(), prompt=prompt)

    def summarize(_: str) -> str:
        docs = store.as_retriever(search_kwargs={"k": 5}).get_relevant_documents("summary")
        full_text = "\n\n".join(d.page_content for d in docs)
        return chain.run({"content": full_text})

    return Tool(name=name, func=summarize, description="Summarize the PDF.")


# ----------------------------------------
# 🧠 LangGraph Agent
# ----------------------------------------
def build_agent(pdf_tool, summary_tool):
    # Simple routing based on keyword
    def agent_logic(input_str: str):
        if "summary" in input_str.lower():
            return summary_tool.run(input_str)
        else:
            return pdf_tool.run(input_str)

    return agent_logic


def build_agents(pdf_tool, summary_tool):
    def router(state):
        if "summary" in state["message"].lower():
            return "summary"
        return "qa"

    builder = StateGraph(dict)
    builder.add_node("qa", lambda state: {"response": pdf_tool.run(state["message"])})
    builder.add_node("summary", lambda state: {"response": summary_tool.run(state["message"])})

    builder.set_entry_point("entry")
    builder.add_conditional_edges("entry", router, {
        "qa": "qa",
        "summary": "summary"
    })

    builder.add_edge("qa", END)
    builder.add_edge("summary", END)

    return builder.compile()


# ----------------------------------------
# 🚀 FastAPI Endpoints
# ----------------------------------------

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_bytes = await file.read()
    collection_name = file.filename.replace(".pdf", "").lower()
    create_or_refresh_store(collection_name, file_bytes)
    return {"status": "uploaded", "collection_name": collection_name}


@app.post("/fin_chat")
async def fin_chat(collection_name: str = Query(...), message: str = Query(...)):
    try:
        store = load_qdrant_store(collection_name)
        pdf_tool = create_pdf_qa_tool(store)
        summary_tool = create_summary_tool(store)
        agent = build_agent(pdf_tool, summary_tool)
        result = agent(message)
        #print(f"{result}")
        return {"response": result}
        #return {"response": result["response"]}
    except Exception as e:
        error_str = traceback.format_exc()
        print("Captured traceback:\n", error_str)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/collection/{collection_name}/all")
def get_collection_data(collection_name: str):
    client = qdrant_client.QdrantClient(url=VECTOR_DB_URL)
    payloads = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            with_payload=True,
            limit=100,
        )
        payloads.extend([r.payload for r in results])
        if offset is None:
            break
    return {"collection": collection_name, "count": len(payloads), "items": payloads}
