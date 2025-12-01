from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
import os
import json
from dotenv import load_dotenv

# LlamaIndex imports for PDF processing and advanced retrieval
from llama_cloud_services import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.schema import TextNode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core import Settings

load_dotenv(override=True)

# ---------------------------
# Configuration
# ---------------------------
PDF_DIR = "ILD_papers"  # Directory containing your PDFs
PARSED_JSON_DIR = "parsed_jsons"
INDEX_DIR = "ild_index"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 250
TOP_K_CHUNKS = 5
COLBERT_TOP_N = 3

# ---------------------------
# 1) PDF Processing & Indexing
# ---------------------------
def azure_embeddings():
    """Create Azure OpenAI embeddings instance."""
    return AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="text-embedding-3-large",
        api_version="2023-05-15",
    )

# Initialize embedding model and ColBERT re-ranker (lazy loading)
_embed_model = None
_colbert_reranker = None

def get_embed_model():
    """Lazy initialization of embedding model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = azure_embeddings()
        Settings.embed_model = _embed_model
    return _embed_model

def get_colbert_reranker():
    """Lazy initialization of ColBERT re-ranker."""
    global _colbert_reranker
    if _colbert_reranker is None:
        _colbert_reranker = ColbertRerank(model="colbert-ir/colbertv2.0", top_n=COLBERT_TOP_N)
    return _colbert_reranker

def parse_pdfs_to_json(pdf_dir: str, out_dir: str):
    """Parse PDFs to JSON using LlamaParse."""
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(pdf_dir):
        print(f"PDF directory {pdf_dir} not found. Skipping PDF parsing.")
        return

    for pdf_file in [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Parsing {pdf_file}...")

        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="json",
            tables_as_text=True,
            do_not_unroll_columns=True,
        )

        try:
            docs = parser.get_json_result(pdf_path)
            for doc in docs:
                pages = doc.get("pages", [])
                if not pages:
                    continue

                out_name = os.path.join(out_dir, f"{os.path.splitext(pdf_file)[0]}_parsed.json")
                with open(out_name, "w", encoding="utf-8") as f:
                    json.dump({"pages": pages}, f, indent=2)
                print(f"✓ Saved {out_name}")
        except Exception as e:
            print(f"Error parsing {pdf_file}: {e}")

def build_index_from_pdfs() -> VectorStoreIndex:
    """Build vector index from parsed PDFs."""
    # Parse PDFs if not already done
    parse_pdfs_to_json(PDF_DIR, PARSED_JSON_DIR)

    # Load parsed JSONs
    json_files = [f for f in os.listdir(PARSED_JSON_DIR) if f.endswith("_parsed.json")]
    if not json_files:
        raise ValueError(f"No parsed JSON files found in {PARSED_JSON_DIR}")

    docstore = SimpleDocumentStore()
    splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    nodes = []

    for json_file in json_files:
        with open(os.path.join(PARSED_JSON_DIR, json_file), encoding="utf-8") as f:
            data = json.load(f)
            pages = data.get("pages", [])
            doc_name = json_file.replace("_parsed.json", "")

            for page in pages:
                page_num = page.get("page", "Unknown")
                content = page.get("md", "") or page.get("text", "")

                if content.strip():
                    for chunk in splitter.split_text(content):
                        nodes.append(
                            TextNode(
                                text=chunk,
                                metadata={
                                    "document": doc_name,
                                    "page": str(page_num)
                                }
                            )
                        )

    print(f"✶ Created {len(nodes)} chunks from PDFs")

    # Build and persist index
    docstore.add_documents(nodes)
    idx = VectorStoreIndex.from_documents(
        nodes,
        storage_context=StorageContext.from_defaults(docstore=docstore),
        embed_model=get_embed_model()
    )
    idx.storage_context.persist(persist_dir=INDEX_DIR)
    return idx

def load_or_build_index() -> VectorStoreIndex:
    """Load existing index or build new one."""
    if os.path.exists(INDEX_DIR):
        print("Loading existing index...")
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=INDEX_DIR))
    else:
        print("Building new index from PDFs...")
        return build_index_from_pdfs()

# Initialize index
vectorstore = load_or_build_index()

# ---------------------------
# 2) Define LangGraph State
# ---------------------------
class State(dict):
    """A simple dict state with:
    - 'question': user question
    - 'docs': retrieved docs
    - 'answer': final answer
    """
    question: str
    docs: list
    answer: str

# -------------------------------------------------------
# 3) Nodes for the graph: retrieve → generate
# -------------------------------------------------------

def retrieve_node(state: State):
    """Retrieve documents using ColBERT re-ranking for better relevance."""
    query = state["question"]

    # Initial retrieval with higher k
    retriever = vectorstore.as_retriever(similarity_top_k=TOP_K_CHUNKS)
    raw_nodes = retriever.retrieve(query)

    # Re-rank with ColBERT
    colbert = get_colbert_reranker()
    reranked_nodes = colbert.postprocess_nodes(raw_nodes, query_str=query)

    # Convert to LangChain Document format and add metadata
    docs = []
    for node in reranked_nodes:
        metadata = node.node.metadata or {}
        doc_text = f"[Document: {metadata.get('document', 'Unknown')}, Page: {metadata.get('page', 'Unknown')}]\n{node.node.text}"
        docs.append(Document(page_content=doc_text, metadata=metadata))

    state["docs"] = docs
    print(f"✓ Retrieved and re-ranked {len(docs)} chunks")
    return state


def generate_node(state: State):
    # LLM‑Initialisierung
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name="gpt-5",
        api_version="2025-03-01-preview",
        temperature=1.0,
        streaming=False,
    )

    docs_text = "\n".join([d.page_content for d in state["docs"]])
    prompt = f"""
                You are a helpful assistant.

                Context:
                {docs_text}

                User question:
                {state['question']}

                Answer using ONLY the context above.
                """
    state["answer"] = llm.invoke(prompt).content
    return state

# ---------------------------
# 4) Build the graph
# ---------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)

graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# ---------------------------
# 5) Run the RAG pipeline
# ---------------------------
if __name__ == "__main__":
    question = "What patterns suggest UIP?"
    result = graph.invoke({"question": question})
    print("ANSWER:\n", result["answer"])