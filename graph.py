import os
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import re
from dotenv import load_dotenv

# LlamaIndex imports for PDF processing and advanced retrieval
from llama_cloud_services import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from rank_bm25 import BM25Okapi

load_dotenv(override=True)

# ---------------------------
# Configuration
# ---------------------------
PDF_DIR = "guidelines"  # Directory containing your PDFs
PARSED_JSON_DIR = "parsed_jsons"
INDEX_DIR = "ild_index"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
TOP_K_CHUNKS = 30
COLBERT_TOP_N = 20
COLBERT_MIN_SCORE = 0.3  # minimum relevance score to keep a chunk
SUB_QUERY_TOP_K = 15      # per-sub-query retrieval count (multi-query mode)
THERAPY_TOP_K = 10        # chunks to retrieve in therapy phase
REWRITE_WORD_THRESHOLD = 30  # legacy fallback (unused — LLM routing replaces this)
CLASSIFY_BATCH_SIZE = 8       # chunks per LLM classification call
CLASSIFY_FILTER_LABELS = {"administrative", "bibliography"}  # discard these

# Pricing per 1M tokens (USD) — adjust to match your Azure deployment
PRICE_INPUT_PER_M = 2.50   # prompt/input tokens
PRICE_OUTPUT_PER_M = 10.00  # completion/output tokens

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
        # Don't set Settings.embed_model to avoid triggering langchain imports
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
        out_name = os.path.join(out_dir, f"{os.path.splitext(pdf_file)[0]}_parsed.json")
        if os.path.exists(out_name):
            print(f"Skipping {pdf_file} (already parsed)")
            continue

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

def _is_bibliography_page(text: str) -> bool:
    """Detect bibliography/reference pages by citation density.

    Matches numbered references like '1. Author ...' or '1  Author ...'
    followed by a year (e.g. 2020). If a page has 5+ such entries it is
    almost certainly a reference list, not clinical content.
    """
    # Pattern: line-start number (with optional dot), then author-like text, then a 4-digit year
    ref_pattern = re.compile(r'^\s*\d{1,3}\.?\s+[A-Z][a-z]+.*(?:19|20)\d{2}', re.MULTILINE)
    matches = ref_pattern.findall(text)
    return len(matches) >= 5


CLASSIFY_SYSTEM_PROMPT = """\
You are a medical document classifier. For each text chunk from a clinical guideline, assign ONE label:

- clinical_recommendation: Treatment guidance, therapeutic decisions, drug dosing, management algorithms
- diagnostic_criteria: Diagnostic criteria, classification systems, differential diagnosis, workup steps
- evidence_summary: Study results, meta-analyses, evidence quality statements
- background: Pathophysiology, epidemiology, definitions, general medical context
- administrative: Author lists, committee protocols, forms, templates, questionnaires, conflict of interest, legal disclaimers, leitlinienreport
- bibliography: Reference lists, numbered citations, literature references

Respond with ONLY a JSON array of labels (one per chunk), e.g.:
["diagnostic_criteria", "clinical_recommendation", "administrative"]"""


def _extract_sections_from_items(pages: list[dict], doc_name: str) -> list[dict]:
    """Walk LlamaParse items across all pages and group text under section headings.

    Returns a list of dicts: {title, text, pages} where pages is a set of page
    numbers the section spans.  Headings that are clearly artifacts (arrows,
    single numbers, figure labels) are ignored and don't start new sections.
    """
    # Pattern to skip artifact headings like "↓", "2.5", "1", "Abb. 2 ..."
    _artifact_re = re.compile(r'^(?:↓|[\d.]+|Abb\.\s*\d+.*)$')

    sections: list[dict] = []
    current_title = "Untitled"
    current_texts: list[str] = []
    current_pages: set[int] = set()

    for page in pages:
        page_num = page.get("page", 0)
        items = page.get("items", [])

        for item in items:
            item_type = item.get("type", "")
            value = (item.get("value") or "").strip()

            if item_type == "heading" and value and not _artifact_re.match(value):
                # Flush previous section
                if current_texts:
                    sections.append({
                        "title": current_title,
                        "text": "\n\n".join(current_texts),
                        "pages": current_pages,
                    })
                current_title = value
                current_texts = []
                current_pages = {page_num}
            elif item_type in ("text", "table"):
                md = item.get("md") or item.get("value") or ""
                md = md.strip()
                if md:
                    current_texts.append(md)
                    current_pages.add(page_num)

    # Flush final section
    if current_texts:
        sections.append({
            "title": current_title,
            "text": "\n\n".join(current_texts),
            "pages": current_pages,
        })

    return sections


def _classify_chunks(chunks: list[TextNode]) -> list[TextNode]:
    """Classify chunks using LLM and filter out non-clinical content.

    Sends chunks in batches to the LLM. Chunks labelled with a category in
    CLASSIFY_FILTER_LABELS are discarded. Surviving chunks get a 'category'
    metadata field.
    """
    if not chunks:
        return chunks

    from langchain_core.messages import SystemMessage as _SM, HumanMessage as _HM

    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name="gpt-5",
        api_version="2025-03-01-preview",
        temperature=1,
    )

    kept = []
    filtered_count = 0

    for batch_start in range(0, len(chunks), CLASSIFY_BATCH_SIZE):
        batch = chunks[batch_start:batch_start + CLASSIFY_BATCH_SIZE]

        # Build the classification request
        batch_text = "\n\n---\n\n".join(
            f"CHUNK {i+1} (section: {node.metadata.get('section', 'N/A')}, "
            f"doc: {node.metadata.get('document', '?')}, p.{node.metadata.get('page', '?')}):\n"
            f"{node.text[:600]}"
            for i, node in enumerate(batch)
        )

        messages = [
            _SM(content=CLASSIFY_SYSTEM_PROMPT),
            _HM(content=f"Classify these {len(batch)} chunks:\n\n{batch_text}"),
        ]

        try:
            response = llm.invoke(messages)
            labels = json.loads(response.content)
            if not isinstance(labels, list) or len(labels) != len(batch):
                # Fallback: keep all if parsing fails
                print(f"  ⚠ Classification parse mismatch (got {len(labels) if isinstance(labels, list) else 'non-list'}), keeping batch")
                for node in batch:
                    node.metadata["category"] = "unknown"
                kept.extend(batch)
                continue
        except (json.JSONDecodeError, Exception) as e:
            print(f"  ⚠ Classification failed ({e}), keeping batch")
            for node in batch:
                node.metadata["category"] = "unknown"
            kept.extend(batch)
            continue

        for node, label in zip(batch, labels):
            label = label.strip().lower()
            node.metadata["category"] = label
            if label in CLASSIFY_FILTER_LABELS:
                filtered_count += 1
            else:
                kept.append(node)

        print(f"  Classified batch {batch_start//CLASSIFY_BATCH_SIZE + 1}: "
              f"{len(batch)} chunks → {sum(1 for n, l in zip(batch, labels) if l.strip().lower() not in CLASSIFY_FILTER_LABELS)} kept")

    print(f"✶ Classification: {len(kept)} kept, {filtered_count} filtered out "
          f"({', '.join(sorted(CLASSIFY_FILTER_LABELS))})")
    return kept


def build_index_from_pdfs() -> VectorStoreIndex:
    """Build vector index from parsed PDFs.

    Pipeline:
      1. Parse PDFs to JSON (LlamaParse, cached)
      2. Extract sections from structured items (section-aware chunking)
      3. Split each section with SentenceSplitter, carrying section title as metadata
      4. Filter bibliography pages (heuristic)
      5. Classify chunks with LLM, discard administrative/bibliography
      6. Build and persist VectorStoreIndex
    """
    # Parse PDFs if not already done
    parse_pdfs_to_json(PDF_DIR, PARSED_JSON_DIR)

    # Load parsed JSONs
    json_files = [f for f in os.listdir(PARSED_JSON_DIR) if f.endswith("_parsed.json")]
    if not json_files:
        raise ValueError(f"No parsed JSON files found in {PARSED_JSON_DIR}")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = []

    for json_file in json_files:
        with open(os.path.join(PARSED_JSON_DIR, json_file), encoding="utf-8") as f:
            data = json.load(f)
            pages = data.get("pages", [])
            doc_name = json_file.replace("_parsed.json", "")

        # Step 1: Extract sections from structured items
        sections = _extract_sections_from_items(pages, doc_name)
        print(f"  {doc_name}: {len(sections)} sections extracted")

        bib_skipped = 0
        for section in sections:
            # Step 2: Skip bibliography sections (heuristic)
            if _is_bibliography_page(section["text"]):
                bib_skipped += 1
                continue

            # Step 3: Chunk within section boundaries
            page_nums = sorted(section["pages"])
            page_str = str(page_nums[0]) if len(page_nums) == 1 else f"{page_nums[0]}-{page_nums[-1]}"

            for chunk_text in splitter.split_text(section["text"]):
                nodes.append(
                    TextNode(
                        text=chunk_text,
                        metadata={
                            "document": doc_name,
                            "page": page_str,
                            "section": section["title"],
                        }
                    )
                )

        if bib_skipped:
            print(f"  ⊘ Skipped {bib_skipped} bibliography sections from {doc_name}")

    print(f"✶ Created {len(nodes)} chunks from PDFs (section-aware)")

    # Step 4: LLM-based classification to filter non-clinical chunks
    nodes = _classify_chunks(nodes)

    # Build and persist index from nodes
    idx = VectorStoreIndex(
        nodes,
        embed_model=get_embed_model()
    )
    idx.storage_context.persist(persist_dir=INDEX_DIR)
    return idx

def _index_is_stale() -> bool:
    """Check if any PDF in PDF_DIR is newer than the persisted index."""
    if not os.path.exists(INDEX_DIR):
        return True
    # Use the index directory's mtime as the baseline
    index_mtime = os.path.getmtime(INDEX_DIR)
    if not os.path.exists(PDF_DIR):
        return False
    for f in os.listdir(PDF_DIR):
        if f.lower().endswith(".pdf"):
            if os.path.getmtime(os.path.join(PDF_DIR, f)) > index_mtime:
                return True
    return False


def _reassemble_vector_store():
    """If the vector store was split into parts for git, reassemble it before loading."""
    target = os.path.join(INDEX_DIR, "default__vector_store.json")
    part1 = target + ".part1"
    if not os.path.exists(part1):
        return  # nothing to reassemble
    if os.path.exists(target):
        return  # already assembled
    print("Reassembling vector store from parts...")
    i = 1
    with open(target, "wb") as out:
        while True:
            part_path = target + f".part{i}"
            if not os.path.exists(part_path):
                break
            with open(part_path, "rb") as part:
                out.write(part.read())
            i += 1
    print(f"  Assembled {i - 1} parts → {os.path.getsize(target) / 1024 / 1024:.1f} MB")


def load_or_build_index() -> VectorStoreIndex:
    """Load existing index or rebuild if PDFs have changed."""
    if os.path.exists(INDEX_DIR) and not _index_is_stale():
        print("Loading existing index...")
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=INDEX_DIR),
            embed_model=get_embed_model()
        )
    else:
        if os.path.exists(INDEX_DIR):
            print("PDFs changed, rebuilding index...")
            import shutil
            shutil.rmtree(INDEX_DIR)
            if os.path.exists(PARSED_JSON_DIR):
                shutil.rmtree(PARSED_JSON_DIR)
        else:
            print("Building new index from PDFs...")
        return build_index_from_pdfs()

# Reassemble split vector store (if deployed via git parts), then load
_reassemble_vector_store()
vectorstore = load_or_build_index()

# ---------------------------
# BM25 keyword index (built from the same nodes as the vector index)
# ---------------------------

def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize with German compound word splitting for BM25.

    German medical texts use compound words (e.g. 'Lungenemphysem',
    'Traktionsbronchiektasen'). Standard whitespace tokenization makes
    BM25 unable to match 'Emphysem' against 'Lungenemphysem'.

    This tokenizer generates overlapping suffixes for long words so that
    sub-word matches work. E.g. 'lungenemphysem' also produces 'emphysem'.
    """
    words = re.findall(r'\w+', text.lower())
    tokens = list(words)
    for w in words:
        if len(w) > 10:
            for i in range(3, len(w) - 4):
                suffix = w[i:]
                if len(suffix) >= 6:
                    tokens.append(suffix)
    return tokens


def _build_bm25_index(vs: VectorStoreIndex):
    """Build a BM25 index from all nodes in the vector store's docstore."""
    all_nodes = list(vs.docstore.docs.values())
    corpus = [_tokenize_for_bm25(node.text) for node in all_nodes]
    bm25 = BM25Okapi(corpus)
    print(f"✓ BM25 index built ({len(all_nodes)} documents)")
    return bm25, all_nodes

_bm25_index, _bm25_nodes = _build_bm25_index(vectorstore)


def _bm25_retrieve(query: str, top_k: int) -> list:
    """Retrieve nodes using BM25 keyword matching.

    Returns a list of NodeWithScore objects (same format as LlamaIndex retrievers)
    so they can be merged with embedding results via RRF.
    """
    tokenized_query = _tokenize_for_bm25(query)
    scores = _bm25_index.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append(NodeWithScore(node=_bm25_nodes[idx], score=float(scores[idx])))
    return results


# ---------------------------
# 2) Define LangGraph State
# ---------------------------
class State(dict):
    """A simple dict state with:
    - 'question': user question
    - 'docs': retrieved docs (diagnostic + therapy combined)
    - 'answer': final answer
    - 'history': list of {"role": "user"|"assistant", "content": "..."} dicts
    - 'is_complex': bool — LLM-determined query complexity (True = full pipeline)
    - 'preliminary_diagnosis': short diagnostic assessment (complex cases only)
    - 'therapy_queries': therapy-specific queries from decomposition
    - 'therapy_docs': therapy docs added in phase 2 (for frontend updates)
    """
    question: str
    docs: list
    answer: str
    history: list
    is_complex: bool
    preliminary_diagnosis: str
    therapy_queries: list
    therapy_docs: list

# -------------------------------------------------------
# 3) Query complexity classification + rewriting
# -------------------------------------------------------

_COMPLEXITY_SYSTEM_PROMPT = """\
You are a medical query router. Classify the user's query as either \
"complex" or "simple".

COMPLEX — requires multi-step diagnostic reasoning:
- Full clinical cases with patient history, lab results, imaging findings
- Questions that need differential diagnosis or treatment planning for a specific patient
- Queries presenting multiple findings that must be synthesized

SIMPLE — can be answered with a single focused retrieval:
- Direct factual questions ("What is the UIP pattern?", "Welche Medikamente bei IPF?")
- Questions about a single guideline recommendation or definition
- Short clarification or follow-up questions

Respond with ONLY the single word "complex" or "simple". Nothing else."""


def _classify_query_complexity(question: str) -> bool:
    """Ask the LLM whether a query needs the complex clinical-case pipeline.

    Returns True for complex cases, False for simple questions.
    """
    llm = _make_llm(streaming=False)
    messages = [
        SystemMessage(content=_COMPLEXITY_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    try:
        response = llm.invoke(messages)
        decision = response.content.strip().lower()
        is_complex = decision.startswith("complex")
        print(f"  Query complexity: {'complex' if is_complex else 'simple'} (LLM: \"{decision}\")")
        return is_complex
    except Exception as e:
        # Fallback to word-count heuristic if the LLM call fails
        fallback = len(question.split()) >= REWRITE_WORD_THRESHOLD
        print(f"  Complexity classification failed ({e}), fallback: {'complex' if fallback else 'simple'}")
        return fallback


REWRITE_SYSTEM_PROMPT = """\
You are an ILD board specialist. Given a clinical case, generate 3-5 \
GUIDELINE WORKFLOW queries — each targeting a DISTINCT diagnostic decision \
point that the ILD guidelines define.

DO NOT describe the patient's symptoms. Instead, ask: "What does the \
guideline say about THIS decision point?"

DIAGNOSTIC DECISION POINTS — generate a query for each that applies:

1. **HRCT pattern classification + diagnostic confidence**: Combine \
pattern criteria with the diagnostic pathway into ONE query. \
Example: "Leitlinie UIP Muster HRCT Kriterien diagnostische Sicherheit \
ohne Biopsie" — covers BOTH the defining HRCT signs AND when a confident \
diagnosis can be made without biopsy.
2. **Mandatory differential diagnoses**: "Leitlinie Differenzialdiagnosen \
Ausschluss bei [pattern] Muster" — which alternative diagnoses MUST be \
excluded per guideline? (e.g., chronic EAA, asbestosis, CTD-ILD for UIP)
3. **Syndrome recognition** (only if findings suggest a combination): \
Look for COMBINATIONS that indicate a specific syndrome:
   - Emphysema + fibrosis → "CPFE combined pulmonary fibrosis emphysema \
Kriterien Diagnostik"
   - Fibrosis + autoimmune → "CTD-ILD Kriterien Diagnostik"
   Skip this if no combination pattern is present.
4. **Occupational / exposure assessment** (only if exposure history \
present): "Leitlinie berufsbedingte ILD [specific exposure] Diagnostik \
Kriterien" — query the guideline criteria for that specific exposure.
5. **Indication for invasive diagnostics**: "Leitlinie Indikation BAL \
Kryobiopsie bei [pattern/context]" — when does the guideline recommend \
BAL or biopsy, and when can it be omitted?

CRITICAL — SEMANTIC DISTINCTNESS:
- Each query MUST target a DIFFERENT section of the guideline.
- Before finalizing, check: would two queries retrieve mostly the same \
chunks? If yes, MERGE them into one.
- BAD example: "UIP Muster HRCT Kriterien" + "IPF Diagnose ohne Biopsie \
bei UIP" → these retrieve the SAME guideline sections. Merge them.
- GOOD example: "UIP Muster HRCT Kriterien diagnostische Sicherheit" + \
"Differenzialdiagnosen Ausschluss bei UIP" → these target DIFFERENT sections.

Rules:
- Generate 3-5 queries. Fewer distinct queries beat more overlapping ones.
- Each query should ask for GUIDELINE CRITERIA, not patient descriptions.
- Frame queries as: "Leitlinie Kriterien für X" / "guideline criteria for X"
- Keep each query under 20 words.
- Mix German and English terms for retrieval coverage.
- Do NOT include therapy/treatment queries — those come later.
- Return ONLY a JSON array of strings, no markdown code blocks."""


def _rewrite_query(question: str, *, is_complex: bool) -> list[str]:
    """Decompose a clinical question into focused diagnostic retrieval queries.

    Returns a list of query strings. Simple questions (is_complex=False)
    are returned as-is.
    For complex cases, ONLY the LLM-generated workflow queries are returned
    (the original question is NOT included — it is too broad and dilutes
    retrieval precision).
    Therapy queries are NOT generated here — they come after the diagnosis.
    """
    if not is_complex:
        return [question]

    llm = _make_llm(streaming=False)
    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=f"Clinical case:\n{question}"),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        # Strip markdown code block if present
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        result = json.loads(content)

        if isinstance(result, list) and all(isinstance(q, str) for q in result):
            queries = [q for q in result[:5] if isinstance(q, str)]
            if queries:
                return queries
    except (json.JSONDecodeError, TypeError, Exception) as e:
        print(f"  Query rewrite failed ({e}), using original question")

    return [question]


# -------------------------------------------------------
# 4) Nodes for the graph: retrieve → generate
# -------------------------------------------------------

def _rrf_merge(ranked_lists: list[list], k: int = 60, top_n: int = COLBERT_TOP_N,
               query_labels: list[str] | None = None):
    """Reciprocal Rank Fusion across multiple ranked node lists.

    For each node, RRF_score = sum(1 / (k + rank)) across all lists where
    it appears. Returns (top_n_nodes, provenance_map) where provenance_map
    maps node_id → set of query labels that contributed to that node.
    """
    scores = {}      # node_id -> cumulative RRF score
    nodes = {}       # node_id -> node object
    provenance = {}  # node_id -> set of query labels

    for list_idx, ranked in enumerate(ranked_lists):
        label = query_labels[list_idx] if query_labels else f"list_{list_idx}"
        for rank, node in enumerate(ranked):
            nid = node.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
            if nid not in nodes:
                nodes[nid] = node
            provenance.setdefault(nid, set()).add(label)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    prov_map = {nid: sorted(provenance[nid]) for nid in sorted_ids}
    return [nodes[nid] for nid in sorted_ids], prov_map


def _nodes_to_docs(nodes) -> list[Document]:
    """Convert LlamaIndex NodeWithScore list to LangChain Documents."""
    docs = []
    for node in nodes:
        metadata = node.node.metadata or {}
        section = metadata.get("section", "")
        section_info = f", Section: {section}" if section else ""
        doc_text = (
            f"[Document: {metadata.get('document', 'Unknown')}, "
            f"Page: {metadata.get('page', 'Unknown')}{section_info}]\n"
            f"{node.node.text}"
        )
        docs.append(Document(page_content=doc_text, metadata=metadata))
    return docs


def _hybrid_retrieve(queries: list[str], top_k_embed: int, top_n: int) -> tuple[list, dict]:
    """Hybrid retrieval: embedding + BM25, merged via RRF.

    For each query, runs both vector search and BM25 keyword search,
    then merges all result lists with Reciprocal Rank Fusion.

    Long queries (>30 words, i.e. full clinical cases) are split into
    sentences for BM25 so that key terms like 'Emphysem' are not diluted
    by hundreds of other tokens. Embedding search uses the full query
    as-is since it handles long input well.

    Returns (merged_nodes, provenance_map) where provenance_map maps
    node_id → list of query strings that contributed to that node.
    """
    ranked_lists = []
    query_labels = []
    retriever = vectorstore.as_retriever(similarity_top_k=top_k_embed)

    for q in queries:
        # Embedding: always use full query (handles long input well)
        ranked_lists.append(retriever.retrieve(q))
        query_labels.append(q)

        # BM25: split long queries into sentences for focused matching
        if len(q.split()) > 30:
            sentences = re.split(r'[.!?\n]+', q)
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) >= 3:
                    ranked_lists.append(_bm25_retrieve(sent, top_k=top_k_embed))
                    query_labels.append(q)  # attribute to parent query
        else:
            ranked_lists.append(_bm25_retrieve(q, top_k=top_k_embed))
            query_labels.append(q)

    return _rrf_merge(ranked_lists, top_n=top_n, query_labels=query_labels)


def retrieve_node(state: State):
    """Retrieve diagnostic documents using hybrid retrieval (embedding + BM25).

    Single query (short questions): hybrid search + ColBERT reranking.
    Multi-query (clinical cases): hybrid search + RRF merging.
    RRF is used instead of ColBERT for multi-query because ColBERT v2
    is English-only and penalizes the German guideline content.
    """
    original_question = state["question"]
    has_history = bool(state.get("history"))

    # Classify query complexity (LLM-based routing)
    if has_history:
        is_complex = False
    else:
        is_complex = _classify_query_complexity(original_question)
    state["is_complex"] = is_complex

    # Step 1: Diagnostic query decomposition
    diagnostic_queries = _rewrite_query(original_question, is_complex=is_complex)
    print(f"  Diagnostic queries ({len(diagnostic_queries)}): {diagnostic_queries}")

    if len(diagnostic_queries) == 1:
        # Single-query path: hybrid retrieval + ColBERT reranking
        hybrid_nodes, _ = _hybrid_retrieve(diagnostic_queries, TOP_K_CHUNKS, TOP_K_CHUNKS)

        colbert = get_colbert_reranker()
        reranked_nodes = colbert.postprocess_nodes(hybrid_nodes, query_str=original_question)

        filtered_nodes = [n for n in reranked_nodes if n.score >= COLBERT_MIN_SCORE]
        if not filtered_nodes:
            filtered_nodes = reranked_nodes[:1] if reranked_nodes else []
    else:
        # Multi-query path: hybrid retrieval + RRF merging
        filtered_nodes, _ = _hybrid_retrieve(diagnostic_queries, SUB_QUERY_TOP_K, COLBERT_TOP_N)

    state["docs"] = _nodes_to_docs(filtered_nodes)
    state["diagnostic_queries"] = diagnostic_queries
    print(f"✓ Retrieved {len(state['docs'])} diagnostic chunks from {len(diagnostic_queries)} queries (hybrid)")
    return state


def _build_provenance_details(provenance: dict, nodes_or_docs, queries: list[str],
                              is_docs: bool = False,
                              node_ids: list[str] | None = None) -> str:
    """Build a per-query breakdown string for step details.

    Args:
        provenance: {node_id: [query_strings]} mapping
        nodes_or_docs: ordered list of NodeWithScore or Document objects
        queries: list of query strings used
        is_docs: True if nodes_or_docs contains Documents (not NodeWithScore)
        node_ids: ordered node_id list matching docs (required when is_docs=True)
    """
    # Invert provenance: query → list of (index, metadata, text) tuples
    query_to_chunks: dict[str, list[tuple[int, dict, str]]] = {}
    for idx, item in enumerate(nodes_or_docs):
        if is_docs:
            nid = node_ids[idx] if node_ids and idx < len(node_ids) else None
            meta = item.metadata
            text = item.page_content
        else:
            nid = item.node.node_id
            meta = item.node.metadata or {}
            text = item.node.text

        # Find which queries contributed to this item
        if nid and nid in provenance:
            contributing_queries = provenance[nid]
        else:
            # Fallback: attribute to all queries
            contributing_queries = queries[:1]

        for q in contributing_queries:
            query_to_chunks.setdefault(q, []).append((idx + 1, meta, text))

    lines = []
    for q_idx, q in enumerate(queries):
        chunks = query_to_chunks.get(q, [])
        label = f"query {q_idx + 1}" if len(queries) > 1 else "query"
        lines.append(f"### Query {q_idx + 1} ({label})")
        lines.append(f"> {q}")
        lines.append("")
        if chunks:
            lines.append(f"Contributed to **{len(chunks)}** chunks:")
            lines.append("")
            for src_idx, meta, text in chunks:
                page = meta.get('page', '?')
                section = meta.get('section', '?')
                lines.append(f"**[{src_idx}] p.{page} | {section}**")
                lines.append("")
                # Indent chunk text as a blockquote so it's collapsible/readable
                for text_line in text.split('\n'):
                    lines.append(f"> {text_line}")
                lines.append("")
        else:
            lines.append("No unique chunks contributed")
        lines.append("")

    return "\n".join(lines)


def stream_retrieve_node(state: State):
    """Generator that yields status events during document retrieval.

    Yields (event_type, content) tuples:
      ("status", text) — progress message for the spinner

    Stores in state: docs, diagnostic_queries, retrieval_provenance, _node_ids.
    """
    original_question = state["question"]
    has_history = bool(state.get("history"))

    # Step 1: Classify query complexity (LLM-based routing)
    if has_history:
        is_complex = False
    else:
        yield ("status", "Classifying query...")
        is_complex = _classify_query_complexity(original_question)
    state["is_complex"] = is_complex

    if is_complex:
        yield ("status", "Decomposing query...")
    diagnostic_queries = _rewrite_query(original_question, is_complex=is_complex)
    print(f"  Diagnostic queries ({len(diagnostic_queries)}): {diagnostic_queries}")

    if len(diagnostic_queries) == 1:
        # Single-query path: hybrid retrieval + ColBERT reranking
        yield ("status", "Searching: hybrid retrieval...")
        hybrid_nodes, prov = _hybrid_retrieve(diagnostic_queries, TOP_K_CHUNKS, TOP_K_CHUNKS)

        yield ("status", "Reranking with ColBERT...")
        colbert = get_colbert_reranker()
        reranked_nodes = colbert.postprocess_nodes(hybrid_nodes, query_str=original_question)

        filtered_nodes = [n for n in reranked_nodes if n.score >= COLBERT_MIN_SCORE]
        if not filtered_nodes:
            filtered_nodes = reranked_nodes[:1] if reranked_nodes else []
        # Rebuild provenance for filtered nodes
        provenance = {n.node.node_id: prov.get(n.node.node_id, []) for n in filtered_nodes}
    else:
        # Multi-query path: per-query status updates + RRF
        total = len(diagnostic_queries)
        ranked_lists = []
        query_labels = []
        retriever = vectorstore.as_retriever(similarity_top_k=SUB_QUERY_TOP_K)

        for q_idx, q in enumerate(diagnostic_queries):
            q_short = q if len(q) <= 60 else q[:57] + "..."
            yield ("status", f"Searching query {q_idx + 1}/{total}: {q_short}")

            # Embedding search
            ranked_lists.append(retriever.retrieve(q))
            query_labels.append(q)

            # BM25 search
            if len(q.split()) > 30:
                sentences = re.split(r'[.!?\n]+', q)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent.split()) >= 3:
                        ranked_lists.append(_bm25_retrieve(sent, top_k=SUB_QUERY_TOP_K))
                        query_labels.append(q)
            else:
                ranked_lists.append(_bm25_retrieve(q, top_k=SUB_QUERY_TOP_K))
                query_labels.append(q)

        yield ("status", "Merging results (RRF)...")
        filtered_nodes, provenance = _rrf_merge(
            ranked_lists, top_n=COLBERT_TOP_N, query_labels=query_labels
        )

    state["docs"] = _nodes_to_docs(filtered_nodes)
    state["diagnostic_queries"] = diagnostic_queries
    state["retrieval_provenance"] = provenance
    state["_node_ids"] = [n.node.node_id for n in filtered_nodes]
    print(f"✓ Retrieved {len(state['docs'])} diagnostic chunks from {len(diagnostic_queries)} queries (hybrid)")


def _build_messages(state: State) -> list:
    """Build a list of LangChain message objects for the LLM.

    Includes: system prompt with retrieved context, prior conversation
    history (if any), and the current user question.
    """
    docs_text = "\n\n".join(
        f"[Source {i+1}] {d.page_content}" for i, d in enumerate(state["docs"])
    )

    is_complex = state.get("is_complex", False)

    if is_complex:
        system_text = (
            "You are an ILD board (multidisciplinary discussion / MDD) simulator. "
            "You ARE the MDD — your output is the board's final consensus. Do NOT "
            "recommend 'presenting the case at an MDD' or 'finalizing the diagnosis "
            "in MDD' — that is what you are doing right now.\n\n"
            "Subspecialties: pulmonology, radiology, rheumatology, pathology, "
            "occupational medicine.\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify the relevant guideline passages from the context below "
            "and cite them inline using [Source N] (where N is the source number).\n"
            "   IMPORTANT: Each source has metadata including its Section title. "
            "Only cite a source if its SECTION is topically relevant to the patient's "
            "case. For example, do not cite a source from a rheumatology section for "
            "a case where autoimmune disease has already been excluded. Prefer sources "
            "from sections that match the clinical question (e.g., IPF sections for "
            "IPF cases, occupational medicine sections for exposure-related cases).\n"
            "2. Based ONLY on those cited passages, provide:\n"
            "   a) Your clinical reasoning and pattern interpretation\n"
            "   b) A definitive diagnosis (or ranked differential with confidence)\n"
            "   c) Recommended workup if gaps remain\n"
            "   d) A concrete treatment plan with specific drugs/doses where the "
            "guidelines provide them\n"
            "3. Every factual claim MUST have at least one [Source N] citation. "
            "Do NOT state any medical facts that are not supported by the provided context.\n"
            "4. If the provided context does not contain enough information to answer "
            "a specific aspect of the question, say so explicitly rather than filling "
            "in from general knowledge.\n"
            "5. At the end of your answer, list all cited sources with their document name "
            "and page number.\n"
            "6. After the source list, add a section '## EXACT_QUOTES' with the exact "
            "verbatim text passages you relied on from each source. Format:\n"
            "   [Source N]: \"exact quote from the source text\"\n"
            "   [Source N]: \"another exact quote from the same or different source\"\n"
            "   Copy the text EXACTLY as it appears in the source — do not paraphrase, "
            "translate, or reorder. These quotes will be used to highlight the relevant "
            "passages in the PDF.\n\n"
            f"Context:\n{docs_text}"
        )
    else:
        system_text = (
            "You are an ILD (interstitial lung disease) specialist answering a "
            "direct medical question. Answer concisely and to the point based ONLY "
            "on the provided guideline passages.\n\n"
            "INSTRUCTIONS:\n"
            "1. Give a clear, focused answer to the question. Do NOT use a rigid "
            "a/b/c/d structure — just answer naturally.\n"
            "2. Cite guideline passages inline using [Source N].\n"
            "3. Every factual claim MUST have at least one [Source N] citation. "
            "Do NOT state medical facts not supported by the provided context.\n"
            "4. If the context does not contain enough information, say so explicitly "
            "rather than filling in from general knowledge.\n"
            "5. At the end, list cited sources with document name and page number.\n"
            "6. After the source list, add a section '## EXACT_QUOTES' with the exact "
            "verbatim text passages you relied on. Format:\n"
            "   [Source N]: \"exact quote from the source text\"\n"
            "   Copy the text EXACTLY as it appears — do not paraphrase, translate, "
            "or reorder.\n\n"
            f"Context:\n{docs_text}"
        )

    messages = [SystemMessage(content=system_text)]

    # Append prior conversation turns
    for turn in state.get("history", []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    # Current question
    messages.append(HumanMessage(content=state["question"]))
    return messages


def _build_summary_messages(state: State) -> list:
    """Build messages for the structured clinical summary (complex cases only)."""
    system_text = (
        "You are an ILD board secretary. Given the detailed MDD discussion below, "
        "write a concise structured clinical summary in German. "
        "Do NOT add new information or citations — only distil what is already in the discussion.\n\n"
        "Output EXACTLY this structure (use the exact headings):\n\n"
        "## Zusammenfassung\n\n"
        "**1. Diagnose**\n"
        "<Most likely diagnosis and key supporting findings in 1-3 sentences>\n\n"
        "**2. Notwendige Folgeuntersuchungen / Differenzialdiagnosen**\n"
        "<Which follow-up investigations are needed to confirm the diagnosis or rule out "
        "differentials. Explicitly state whether a biopsy is indicated (Biopsie: ja/nein) "
        "and why.>\n\n"
        "**3. Prozedere**\n"
        "<Concrete next steps: referrals, therapy initiation, monitoring, follow-up intervals.>\n\n"
        "Keep each section to 2-4 sentences. Do not repeat lengthy source citations — "
        "refer to findings by name only."
    )
    return [
        SystemMessage(content=system_text),
        HumanMessage(content=(
            f"Detailed MDD discussion:\n\n{state['answer']}\n\n"
            f"Original patient question:\n{state['question']}"
        )),
    ]


def _make_llm(streaming=False):
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name="gpt-5",
        api_version="2025-03-01-preview",
        temperature=1,
        streaming=streaming,
    )


def generate_node(state: State):
    llm = _make_llm(streaming=False)
    state["answer"] = llm.invoke(_build_messages(state)).content
    return state


def _build_verify_messages(state: State) -> list:
    """Build messages for the citation verification step.

    Two-directional verification:
      1. Claim → Source: Is the cited source actually saying what the claim says?
      2. Claim → Patient: Does the claim accurately reflect the patient's findings
         from the original question, or were they silently replaced by generic
         guideline descriptions?
    """
    docs_text = "\n\n".join(
        f"[Source {i+1}] {d.page_content}" for i, d in enumerate(state["docs"])
    )
    return [
        SystemMessage(content=(
            "You are a medical citation verification assistant. You perform TWO checks "
            "on every claim in the answer:\n\n"
            "CHECK 1 — Claim vs. Source (citation accuracy):\n"
            "For each [Source N] citation, verify that the claim it supports is actually "
            "present in or clearly implied by Source N.\n\n"
            "CHECK 2 — Claim vs. Patient (clinical accuracy):\n"
            "Compare every clinical statement in the answer against the ORIGINAL PATIENT "
            "QUESTION below. Watch for cases where the answer silently replaces patient-"
            "specific findings with generic guideline descriptions. For example:\n"
            "  - Patient report says 'dorsal betont' but answer says 'basal/peripher "
            "betont' citing a guideline → the guideline description was substituted for "
            "the actual patient finding.\n"
            "  - Patient has a specific distribution pattern but the answer describes "
            "the textbook pattern from the guideline instead.\n"
            "The answer must clearly distinguish between what the GUIDELINE says (general "
            "criteria) and what the PATIENT actually has (specific findings). If they "
            "differ, the answer must note the discrepancy rather than silently adopting "
            "the guideline wording as if it were the patient's finding.\n\n"
            "ACTIONS:\n"
            "1. Remove or correct any citations where the source does not support the claim.\n"
            "2. Where the answer misrepresents patient findings by substituting guideline "
            "language, correct the statement to accurately reflect the patient's actual "
            "findings and note any discrepancy with the guideline criteria.\n"
            "3. Remove any factual claims that have no valid citation after your review.\n"
            "4. Keep the rest of the answer exactly as-is — do not rephrase, expand, or "
            "add new information beyond the corrections above.\n"
            "5. Return ONLY the corrected answer, nothing else.\n\n"
            f"Sources:\n{docs_text}"
        )),
        HumanMessage(content=(
            f"ORIGINAL PATIENT QUESTION:\n{state['question']}\n\n"
            f"ANSWER TO VERIFY:\n{state['answer']}"
        )),
    ]


def verify_node(state: State):
    """Verify citations via LLM two-directional check."""
    llm = _make_llm(streaming=False)
    verified = llm.invoke(_build_verify_messages(state)).content
    clean_answer, _ = _extract_exact_quotes(verified)
    state["answer"] = clean_answer
    return state


# -------------------------------------------------------
# 5) Two-phase retrieval: diagnose → retrieve therapy
# -------------------------------------------------------

DIAGNOSE_SYSTEM_PROMPT = """\
You are an ILD (interstitial lung disease) specialist. Based on the clinical case \
and the retrieved guideline passages, provide:

1. A SHORT preliminary assessment (3-5 sentences):
   - The most likely ILD diagnosis/subtype
   - Key supporting findings from the case
   - Important differential diagnoses to consider
   - What additional information (especially regarding therapy) is needed
   Be specific and concise. Do NOT provide treatment recommendations yet.

2. A list of RELEVANT source numbers: which of the provided sources are actually \
relevant to THIS specific patient's case. Exclude sources from disease sections that \
do not apply (e.g. rheumatology sections for a non-autoimmune case, pneumoconiosis \
sections for a case without occupational exposure, etc.). \
Only include sources whose SECTION and CONTENT are topically relevant.

Format your response EXACTLY as:
ASSESSMENT: <your assessment text>
RELEVANT_SOURCES: [1, 3, 5, 7]

The RELEVANT_SOURCES line must be a JSON array of source numbers (1-based). \
Do NOT include any other text after the array.

Guideline passages:
{docs_text}"""

THERAPY_QUERY_PROMPT = """\
You are a medical information retrieval specialist. Given a clinical case and a \
preliminary diagnostic assessment, generate 2-4 SHORT search queries to retrieve \
TREATMENT and MANAGEMENT guidelines from ILD clinical practice guidelines.

Rules:
- Focus on: therapy options, drug dosing, management algorithms, progression \
criteria, monitoring, prognosis, follow-up
- Use the specific diagnosis from the assessment to target the right sections
- Include queries in both German and English for retrieval coverage
- Keep each query under 20 words
- If the original question asks about specific aspects not yet covered (e.g., \
prognosis, surgical options, specific drugs), include targeted queries for those
- Return ONLY a JSON array of query strings, no markdown code blocks"""


def _build_diagnose_messages(state: State) -> list:
    """Build messages for the preliminary diagnosis step."""
    docs_text = "\n\n".join(
        f"[Source {i+1}] {d.page_content}" for i, d in enumerate(state["docs"])
    )
    return [
        SystemMessage(content=DIAGNOSE_SYSTEM_PROMPT.format(docs_text=docs_text)),
        HumanMessage(content=state["question"]),
    ]


def _parse_diagnose_response(response_text: str) -> tuple[str, list[int]]:
    """Parse the diagnose node response into assessment + relevant source indices.

    Returns (assessment_text, list_of_1based_source_numbers).
    Falls back gracefully if parsing fails.
    """
    assessment = response_text
    relevant = []

    # Try to extract RELEVANT_SOURCES line
    match = re.search(r'RELEVANT_SOURCES:\s*\[([^\]]*)\]', response_text)
    if match:
        try:
            relevant = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        except ValueError:
            pass
        # Extract ASSESSMENT text (everything between ASSESSMENT: and RELEVANT_SOURCES:)
        assess_match = re.search(r'ASSESSMENT:\s*(.*?)(?=\nRELEVANT_SOURCES:)', response_text, re.DOTALL)
        if assess_match:
            assessment = assess_match.group(1).strip()

    return assessment, relevant


def diagnose_node(state: State):
    """Generate a preliminary diagnosis and filter irrelevant sources.

    Only runs for complex clinical cases (is_complex=True).
    Simple questions skip this step.

    After diagnosis, sources from irrelevant guideline sections are removed
    to avoid wasting context slots on the generation step.
    """
    if not state.get("is_complex", False):
        return state

    llm = _make_llm(streaming=False)
    response = llm.invoke(_build_diagnose_messages(state))

    assessment, relevant_indices = _parse_diagnose_response(response.content)
    state["preliminary_diagnosis"] = assessment
    print(f"✓ Preliminary diagnosis: {assessment[:120]}...")

    # Filter docs to only relevant sources (if the LLM provided a valid list)
    if relevant_indices:
        original_count = len(state["docs"])
        print(f"  RELEVANT_SOURCES: {relevant_indices}")

        # Rescue dropped sources that contain key abbreviations from assessment
        rescued = set()
        for i, doc in enumerate(state["docs"], 1):
            if i not in relevant_indices:
                text_lower = doc.page_content.lower()
                for term in re.findall(r'\b[A-Z]{2,}\b', assessment):
                    if term.lower() in text_lower:
                        rescued.add(i)
                        break

        keep_indices = set(relevant_indices) | rescued

        for i, doc in enumerate(state["docs"], 1):
            if i in rescued:
                status = "⟳ RESCUE"
            elif i in relevant_indices:
                status = "✓ KEEP"
            else:
                status = "✗ DROP"
            meta = doc.metadata
            print(f"    [{i:2d}] {status} | p.{meta.get('page','?')} | {meta.get('section','?')[:60]}")

        filtered = [
            state["docs"][i - 1]
            for i in sorted(keep_indices)
            if 1 <= i <= len(state["docs"])
        ]
        if filtered:
            state["docs"] = filtered
            removed = original_count - len(filtered)
            if removed > 0:
                print(f"  ✂ Filtered {removed} irrelevant sources → {len(filtered)} remaining")
        else:
            print("  ⚠ All sources would be filtered — keeping originals")
    else:
        print("  ⚠ No RELEVANT_SOURCES parsed — keeping all sources")

    return state


def retrieve_therapy_node(state: State):
    """Retrieve therapy-specific chunks based on preliminary diagnosis.

    Combines stored therapy queries from the initial decomposition with
    new queries generated from the preliminary diagnosis (gap detection).
    Deduplicates against existing docs and merges into state["docs"].
    Only runs for complex clinical cases.
    """
    if not state.get("is_complex", False):
        return state

    diagnosis = state.get("preliminary_diagnosis", "")
    therapy_queries = []

    # Generate therapy queries based on the actual diagnosis (gap detection)
    if diagnosis:
        llm = _make_llm(streaming=False)
        messages = [
            SystemMessage(content=THERAPY_QUERY_PROMPT),
            HumanMessage(content=(
                f"Clinical case:\n{state['question']}\n\n"
                f"Preliminary assessment:\n{diagnosis}"
            )),
        ]
        try:
            response = llm.invoke(messages)
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            extra_queries = json.loads(content)
            if isinstance(extra_queries, list):
                therapy_queries.extend(
                    q for q in extra_queries[:4] if isinstance(q, str)
                )
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Therapy query generation failed ({e})")

    if not therapy_queries:
        print("  No therapy queries — skipping therapy retrieval")
        return state

    state["therapy_queries"] = therapy_queries
    print(f"  Therapy queries ({len(therapy_queries)}): {therapy_queries}")

    # Retrieve therapy chunks via hybrid retrieval (embedding + BM25)
    therapy_nodes, therapy_prov = _hybrid_retrieve(therapy_queries, THERAPY_TOP_K, THERAPY_TOP_K)

    # Deduplicate against existing docs
    existing_texts = {d.page_content for d in state["docs"]}
    new_docs = []
    new_node_ids = []
    for node in therapy_nodes:
        doc = _nodes_to_docs([node])[0]
        if doc.page_content not in existing_texts:
            new_docs.append(doc)
            new_node_ids.append(node.node.node_id)

    state["therapy_docs"] = new_docs
    state["therapy_provenance"] = {nid: therapy_prov.get(nid, []) for nid in new_node_ids}
    state["_therapy_node_ids"] = new_node_ids
    state["docs"].extend(new_docs)
    print(f"✓ Added {len(new_docs)} therapy chunks (total: {len(state['docs'])})")
    return state


def _extract_exact_quotes(answer: str) -> tuple[str, dict[int, list[str]]]:
    """Extract the EXACT_QUOTES section from the answer.

    Returns (answer_without_quotes, {source_index (0-based): [verbatim quotes]}).
    The quotes section is stripped from the displayed answer.
    """
    quotes: dict[int, list[str]] = {}

    # Split off the EXACT_QUOTES section
    parts = re.split(r'##\s*EXACT_QUOTES\s*\n?', answer, maxsplit=1)
    clean_answer = parts[0].rstrip()

    if len(parts) < 2:
        return clean_answer, quotes

    quotes_section = parts[1]
    # Parse lines like: [Source 3]: "exact quote text here"
    for match in re.finditer(
        r'\[Source\s+(\d+)\]\s*:\s*"([^"]+)"', quotes_section
    ):
        idx = int(match.group(1)) - 1  # 0-based
        quote = match.group(2).strip()
        if len(quote) < 5:
            continue
        if idx not in quotes:
            quotes[idx] = []
        if quote not in quotes[idx]:
            quotes[idx].append(quote)

    # # Only keep quotes for sources actually cited in the answer text
    # cited_sources = {int(m.group(1)) - 1 for m in re.finditer(r'\[Source\s+(\d+)\]', clean_answer)}
    # quotes = {idx: qs for idx, qs in quotes.items() if idx in cited_sources}

    return clean_answer, quotes




def stream_generate(state: State):
    """Generator that yields (event_type, content) tuples.

    For complex clinical cases (is_complex=True, determined by LLM routing):
      1. Diagnose → preliminary assessment from diagnostic docs
      2. Retrieve therapy → additional therapy-specific chunks
      3. Generate → answer using all sources
      4. Verify → citation check

    For short questions: Generate → Verify only.

    Event types yielded:
      ("status", text)           — progress message for the spinner
      ("sources_update", list)   — additional therapy source dicts for frontend
      ("token", word)            — answer word for streaming display
      ("replace", full_answer)   — corrected answer if verify changed it
    """
    llm = _make_llm(streaming=False)
    is_complex = state.get("is_complex", False)
    all_responses = []  # track LLM responses for usage aggregation

    if is_complex:
        # Phase 1: Preliminary diagnosis + source relevance filtering
        yield ("status", "Analyzing clinical case...")
        diagnose_response = llm.invoke(_build_diagnose_messages(state))
        all_responses.append(diagnose_response)

        assessment, relevant_indices = _parse_diagnose_response(diagnose_response.content)
        state["preliminary_diagnosis"] = assessment
        print(f"✓ Preliminary diagnosis: {assessment[:120]}...")

        # Step: Preliminary Diagnosis
        diag_summary = assessment.split('\n')[0][:120] if assessment else "Assessment generated"
        yield ("step", {
            "title": "Preliminary Diagnosis",
            "summary": diag_summary,
            "details": assessment,
        })

        # Filter irrelevant sources
        if relevant_indices:
            original_count = len(state["docs"])
            all_docs_before_filter = list(state["docs"])
            print(f"  RELEVANT_SOURCES: {relevant_indices}")

            # Rescue: dropped sources that contain key terms from the
            # assessment should be kept (the LLM recognised a concept
            # but didn't realise the source supports it).
            rescued = set()
            for i, doc in enumerate(state["docs"], 1):
                if i not in relevant_indices:
                    text_lower = doc.page_content.lower()
                    # Check for terms the assessment explicitly mentions
                    for term in re.findall(r'\b[A-Z]{2,}\b', assessment):
                        # Match abbreviations like CPFE, NSIP, CTD, EAA etc.
                        if term.lower() in text_lower:
                            rescued.add(i)
                            break

            keep_indices = set(relevant_indices) | rescued

            for i, doc in enumerate(state["docs"], 1):
                if i in rescued:
                    status = "⟳ RESCUE"
                elif i in relevant_indices:
                    status = "✓ KEEP"
                else:
                    status = "✗ DROP"
                meta = doc.metadata
                print(f"    [{i:2d}] {status} | p.{meta.get('page','?')} | {meta.get('section','?')[:60]}")

            filtered = [
                state["docs"][i - 1]
                for i in sorted(keep_indices)
                if 1 <= i <= len(state["docs"])
            ]
            if filtered and len(filtered) < original_count:
                state["docs"] = filtered
                removed = original_count - len(filtered)
                print(f"  ✂ Filtered {removed} irrelevant sources → {len(filtered)} remaining")

                # Step: Source Filtering
                kept_count = len(set(relevant_indices) & keep_indices)
                rescued_count = len(rescued)
                dropped_count = original_count - len(filtered)
                filter_lines = []
                for i, doc in enumerate(all_docs_before_filter, 1):
                    meta = doc.metadata
                    if i in rescued:
                        tag = "RESCUE"
                    elif i in relevant_indices:
                        tag = "KEEP"
                    else:
                        tag = "DROP"
                    filter_lines.append(
                        f"- [{i}] **{tag}** — p.{meta.get('page','?')} | {meta.get('section','?')[:60]}"
                    )
                yield ("step", {
                    "title": "Source Filtering",
                    "summary": f"{kept_count} kept, {rescued_count} rescued, {dropped_count} dropped",
                    "details": "\n".join(filter_lines),
                })

                # Tell frontend to replace source list with only relevant ones
                yield ("sources_replace", [
                    {
                        "document": doc.metadata.get("document", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "text": doc.page_content,
                    }
                    for doc in state["docs"]
                ])

        # Phase 2: Therapy retrieval (includes gap-detection query generation)
        yield ("status", "Retrieving therapy guidelines...")
        pre_therapy_count = len(state["docs"])
        retrieve_therapy_node(state)

        # Step: Therapy Queries
        therapy_queries = state.get("therapy_queries", [])
        if therapy_queries:
            yield ("step", {
                "title": "Therapy Queries",
                "summary": f"Generated {len(therapy_queries)} therapy queries",
                "details": "\n".join(f"- {q}" for q in therapy_queries),
            })

        # Send new therapy sources to the frontend
        therapy_docs = state.get("therapy_docs", [])
        if therapy_docs:
            therapy_sources = [
                {
                    "document": doc.metadata.get("document", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "text": doc.page_content,
                }
                for doc in therapy_docs
            ]
            yield ("sources_update", therapy_sources)

            # Step: Therapy Retrieval — with per-query breakdown
            therapy_prov = state.get("therapy_provenance", {})
            therapy_node_ids = state.get("_therapy_node_ids", [])
            if therapy_prov and therapy_queries:
                # Build provenance details using therapy docs
                therapy_details = (
                    f"**Total chunks:** {len(therapy_docs)}\n\n"
                    + _build_provenance_details(
                        therapy_prov, therapy_docs, therapy_queries,
                        is_docs=True, node_ids=therapy_node_ids,
                    )
                )
            else:
                therapy_details = "\n".join(
                    f"- p.{d.metadata.get('page','?')} | {d.metadata.get('section','?')[:60]}"
                    for d in therapy_docs
                )
            yield ("step", {
                "title": "Therapy Retrieval",
                "summary": f"Added {len(therapy_docs)} therapy chunks (total: {len(state['docs'])})",
                "details": therapy_details,
            })

    # Phase 3: Generate answer
    if is_complex:
        yield ("status", "Generating answer...")
    gen_response = llm.invoke(_build_messages(state))

    # Extract exact quotes and strip them from the displayed answer
    clean_answer, exact_quotes = _extract_exact_quotes(gen_response.content)
    state["answer"] = clean_answer
    all_responses.append(gen_response)

    for word in state["answer"].split(" "):
        yield ("token", word + " ")

    # Phase 4: Verify citations
    yield ("status", "Verifying citations...")
    generated_answer = state["answer"]
    verify_response = llm.invoke(_build_verify_messages(state))

    # Strip any EXACT_QUOTES the verify node might echo back
    verified_answer, verify_quotes = _extract_exact_quotes(verify_response.content)
    state["answer"] = verified_answer
    all_responses.append(verify_response)

    # Merge quotes from both generate and verify (generate takes priority)
    for idx, quotes in verify_quotes.items():
        if idx not in exact_quotes:
            exact_quotes[idx] = quotes

    verify_changed = state["answer"] != generated_answer
    if verify_changed:
        yield ("replace", state["answer"])

    # Step: Citation Verification
    yield ("step", {
        "title": "Citation Verification",
        "summary": "Answer corrected" if verify_changed else "Answer unchanged",
        "details": "The LLM verify step corrected citation issues in the answer." if verify_changed else "All citations passed LLM verification — no changes needed.",
    })

    # Step: Exact Quotes (debug visibility)
    if exact_quotes:
        quote_lines = []
        for src_idx in sorted(exact_quotes):
            for q in exact_quotes[src_idx]:
                quote_lines.append(f"- **[Source {src_idx + 1}]**: \"{q}\"")
        yield ("step", {
            "title": "Exact Quotes",
            "summary": f"{sum(len(qs) for qs in exact_quotes.values())} quotes from {len(exact_quotes)} sources",
            "details": "\n".join(quote_lines),
        })

        # Send exact quotes for precise PDF highlighting
        yield ("highlights", exact_quotes)

    # Phase 5 (complex only): Structured clinical summary
    if is_complex:
        yield ("status", "Generating clinical summary...")
        summary_response = llm.invoke(_build_summary_messages(state))
        all_responses.append(summary_response)
        summary_text = summary_response.content.strip()
        # Append summary to the displayed answer
        separator = "\n\n---\n\n"
        yield ("token", separator)
        for word in summary_text.split(" "):
            yield ("token", word + " ")
        state["answer"] += separator + summary_text

    # Aggregate token usage from all LLM calls
    input_tokens = sum(
        (r.usage_metadata or {}).get("input_tokens", 0) for r in all_responses
    )
    output_tokens = sum(
        (r.usage_metadata or {}).get("output_tokens", 0) for r in all_responses
    )
    cost = (input_tokens * PRICE_INPUT_PER_M + output_tokens * PRICE_OUTPUT_PER_M) / 1_000_000
    state["usage"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost, 6),
    }

# ---------------------------
# 6) Build the graph
# ---------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("diagnose", diagnose_node)
graph_builder.add_node("retrieve_therapy", retrieve_therapy_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_node("verify", verify_node)

graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "diagnose")
graph_builder.add_edge("diagnose", "retrieve_therapy")
graph_builder.add_edge("retrieve_therapy", "generate")
graph_builder.add_edge("generate", "verify")
graph_builder.add_edge("verify", END)

graph = graph_builder.compile()

# ---------------------------
# 7) Run the RAG pipeline
# ---------------------------
if __name__ == "__main__":
    question = "Should combination therapy versus control be used for any patient with CTD-ILD?"
    result = graph.invoke({"question": question})
    print("ANSWER:\n", result["answer"])