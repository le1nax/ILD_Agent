# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) system for answering medical questions about Interstitial Lung Disease (ILD) — including all subtypes such as CTD-ILD, IPF, hypersensitivity pneumonitis, etc. — using clinical practice guidelines. Built with LangGraph, LlamaIndex, and Azure OpenAI. Includes a Flask web UI with streaming responses.

## Environment Setup

```bash
# conda must be initialized first (not on PATH by default)
source /home/homesOnMaster/dgeiger/miniforge3/etc/profile.d/conda.sh
conda activate ild
# Python env: /home/homesOnMaster/dgeiger/miniforge3/envs/ild (Python 3.12.3)
```

Key dependencies (no `requirements.txt` exists): `langgraph`, `langchain-openai`, `llama-index`, `llama-cloud-services`, `llama-index-embeddings-azure-openai`, `llama-index-postprocessor-colbert-rerank`, `flask`, `python-dotenv`.

## Running

```bash
# Web app (Flask, port 5000)
python app.py

# CLI one-shot query
python graph.py
```

No tests, linting, or build steps are configured.

## Architecture

### graph.py — RAG Pipeline

The entire RAG pipeline in a single file, structured as a five-node LangGraph workflow:

1. **Retrieve node** (`retrieve_node`): Diagnostic document retrieval using categorized query decomposition.
   - **Short questions** (<30 words): Single vector search → ColBERT reranking (`colbert-ir/colbertv2.0`) → filter by `COLBERT_MIN_SCORE`.
   - **Long clinical cases** (>=30 words): `_rewrite_query` decomposes into categorized queries (`{"diagnostic": [...], "therapy": [...]}`) via LLM. Diagnostic queries get per-query vector search + Reciprocal Rank Fusion (`_rrf_merge`). Therapy queries are stored in state for later. RRF is used instead of ColBERT for multi-query because ColBERT v2 is English-only and would penalize the German guideline content.
   Bibliography pages are filtered out during indexing (`_is_bibliography_page`).
2. **Diagnose node** (`diagnose_node`): For complex cases only. Uses retrieved diagnostic docs to generate a short preliminary assessment (most likely diagnosis, key findings, differentials, info gaps). Also identifies which sources are relevant to the specific case and **filters out irrelevant sources** (e.g., rheumatology sections for an IPF case). Sends a `sources_replace` event to update the frontend. Skipped for short questions.
3. **Retrieve therapy node** (`retrieve_therapy_node`): For complex cases only. Combines therapy queries from initial decomposition with new diagnosis-specific queries generated via LLM (gap detection). Retrieves therapy chunks via RRF, deduplicates against existing docs, merges into `state["docs"]`. Skipped for short questions.
4. **Generate node** (`generate_node`): Feeds all retrieved context (diagnostic + therapy) + conversation history into Azure GPT-5 (`deployment_name="gpt-5"`) via LangChain's `AzureChatOpenAI`. The system prompt enforces inline `[Source N]` citations for every factual claim.
5. **Verify node** (`verify_node`): Two-directional citation verification: (a) Claim→Source — checks each `[Source N]` is supported by the source text; (b) Claim→Patient — checks the answer doesn't silently replace patient-specific findings with generic guideline descriptions. Followed by **deterministic citation validation** (`_deterministic_citation_check`) — a purely programmatic (no LLM) check that removes citations where: (i) EXACT_QUOTES attributed to a source don't actually appear in its text (fuzzy substring match), or (ii) the medical abbreviation nearest to a `[Source N]` marker doesn't appear in Source N's text. This catches hallucinated citations that the LLM-based verify misses.

**Important:** `vectorstore = load_or_build_index()` runs at module import time (line ~371). Importing `graph.py` triggers index loading (or rebuild), which takes significant time on first run.

`State` is a `dict` subclass with keys: `question`, `docs`, `answer`, `history`, `preliminary_diagnosis`, `therapy_queries`, `therapy_docs`, and optionally `usage`.

`stream_generate` is a separate generator used by the web app — for complex cases it runs diagnose (+ source filtering) → retrieve_therapy → generate → verify → deterministic citation check; for short questions just generate → verify → deterministic citation check. It yields `("status", text)` for progress updates, `("sources_replace", list)` when irrelevant sources are filtered out after diagnosis, `("sources_update", list)` when therapy sources arrive, `("token", word)` for simulated streaming, `("replace", full_answer)` if verify changes the answer, and `("highlights", dict)` with exact verbatim quotes for PDF highlighting. Aggregates token usage from all LLM calls in `state["usage"]`.

**Data pipeline** (runs once, results cached to disk):
```
PDFs in guidelines/
  -> LlamaParse (PDF -> JSON, stored in parsed_jsons/)
  -> Section-aware chunking (_extract_sections_from_items: groups items under headings)
  -> SentenceSplitter (1024 tokens, 256 overlap) within each section
  -> Bibliography filter (heuristic: _is_bibliography_page)
  -> LLM classification (_classify_chunks: batches of 8, filters administrative/bibliography)
  -> VectorStoreIndex (persisted in ild_index/)
```

Each chunk carries metadata: `document`, `page` (range if section spans pages), `section` (heading title), and `category` (LLM-assigned label: clinical_recommendation, diagnostic_criteria, evidence_summary, background).

The index auto-rebuilds when any PDF in `guidelines/` is newer than the persisted index (`_index_is_stale()`). Delete `ild_index/` to force a full rebuild. Rebuilding triggers LLM classification calls (~63 batches of 8 chunks each).

### app.py — Flask Web UI

- `POST /query` — SSE endpoint. Receives `{question, session_id}`, streams events in order: `session` → `sources` (diagnostic) → `status` → optionally `sources_replace` (filtered) → optionally `sources_update` (therapy) → `token` (repeated) → optionally `replace` (if verify changed) → `highlights` (exact quotes) → `usage` → `done`.
- `POST /new_chat` — clears server-side session history.
- Sessions stored in-memory (`_sessions` dict); lost on restart.
- Tracks per-query cost via both local token counting (`state["usage"]`) and a remote proxy usage API (`/usage/user/me` at `USAGE_API_URL`).

### Frontend

- `templates/index.html` — single-page chat UI with inline CSS
- `static/app.js` — SSE client, renders markdown via `marked.js`, collapsible sources with expandable chunk text

### test_cases.yaml

10 clinical cases (history + radiology report + expected diagnosis) for manual evaluation. Not wired to any automated test runner.

## Key Configuration (top of graph.py)

- `PDF_DIR`, `PARSED_JSON_DIR`, `INDEX_DIR` — directory paths
- `CHUNK_SIZE=1024`, `CHUNK_OVERLAP=256` — sentence-aware text splitting
- `TOP_K_CHUNKS=30` — initial vector retrieval count
- `COLBERT_TOP_N=20` — how many chunks survive reranking (single-query path)
- `COLBERT_MIN_SCORE=0.3` — minimum relevance score to keep a chunk
- `SUB_QUERY_TOP_K=15` — per-sub-query retrieval count (multi-query path)
- `THERAPY_TOP_K=10` — chunks to retrieve in therapy phase
- `REWRITE_WORD_THRESHOLD=30` — questions below this word count skip query decomposition and the diagnose/therapy phases
- `CLASSIFY_BATCH_SIZE=8` — chunks per LLM classification call during indexing
- `CLASSIFY_FILTER_LABELS={"administrative", "bibliography"}` — chunk categories to discard
- `PRICE_INPUT_PER_M`, `PRICE_OUTPUT_PER_M` — token pricing for cost estimates

## Known Limitations & Problems

### Retrieval quality

- **Vector similarity ≠ clinical relevance.** The retrieval is purely embedding-based. A blank ILD Board protocol template that mentions "antifibrotische Therapie mit ___" scores similarly to an actual guideline paragraph explaining when to use antifibrotic therapy. There is no concept of whether a chunk contains actionable clinical content.
- **Non-clinical content filtering (partially addressed).** LLM-based chunk classification now filters out administrative content (protocol templates, forms, questionnaires) and bibliography during indexing. This is more robust than manual heuristics but adds LLM cost during index builds and depends on classification accuracy.
- **Near-duplicate chunks are not deduplicated.** The ILD Board protocol template appears on both page 47 and page 75 of the AWMF guideline with nearly identical text, consuming 2 of 6 source slots with the same non-clinical content.
- **Chunk count is more generous but still finite.** `COLBERT_TOP_N=20` for diagnostic retrieval + up to `THERAPY_TOP_K=10` therapy chunks. Two-phase retrieval separates diagnostic and treatment needs.
- **ColBERT and RRF reranking don't help with content quality.** They optimize for query-document textual relevance, not for whether a chunk contains a clinical recommendation vs. a form field.

### Clinical reasoning

- **Two-phase retrieval partially addresses diagnostic/therapy separation.** Complex cases now get a preliminary diagnosis step, then targeted therapy retrieval. However, the preliminary diagnosis is only as good as the initial diagnostic retrieval.
- **Structured query decomposition (partially addressed).** `_rewrite_query` now returns categorized queries (`diagnostic` and `therapy`), targeting HRCT patterns, differentials, and case-specific findings separately. But the categorization is still LLM-generated and may not always align with clinical reasoning steps.
- **Gap detection (partially addressed).** The therapy query generation step considers the preliminary diagnosis and original question to identify missing aspects. However, it cannot detect when guidelines simply don't cover a condition well.
- **Guideline coverage gaps still cause silent failures.** When the guidelines don't cover a condition well (e.g., CPFE, drug-induced ILD), the system defaults to the closest match it can find rather than flagging insufficient evidence.

### Citation / verification

- **Verify node now does two-directional checking (partially addressed).** CHECK 1 verifies Claim→Source accuracy. CHECK 2 verifies Claim→Patient fidelity (catches when patient findings are silently replaced by guideline descriptions). However, it still cannot judge whether a source truly makes a clinical statement vs. being an administrative form.
- **Deterministic citation validation (partially addressed).** `_deterministic_citation_check` runs after the LLM verify step and programmatically removes citations where: (a) EXACT_QUOTES attributed to a source don't appear in it (fuzzy substring match), or (b) NONE of the medical abbreviations (2+ uppercase letters) from the 200-char context before a citation appear in the source text. This catches clear misattributions (e.g., citing Source N for CPFE when Source N only covers UIP/NSIP) while allowing citations where the source has related terms (e.g., a CT analysis source cited for IPF, where the source contains HRCT). Operates per-citation-instance, not per-source — the same source can be validly cited in one sentence and invalidly in another.
- **Remaining gap: semantic misattribution with shared terms.** The deterministic check relies on uppercase abbreviations as term signals. If a citing sentence shares at least one abbreviation with the source, the citation passes even if the source doesn't actually support the specific claim. The LLM verify step covers some of these cases.
- **The system can produce confidently cited but wrong answers.** Case 10 achieved 100% citation faithfulness (34/34 supported) while giving the wrong diagnosis — the citations were faithful to the sources, but the sources were wrong for the case.

### Evaluation results (10 test cases, eval.py)

- **Diagnosis accuracy:** 10/10 match (after section-aware chunking + LLM classification + 20 chunks).
- **Citation faithfulness:** ~95% auto-judged (310/326 supported, 16 unsupported, 2 uncited). Auto-judge may miss citations to administrative templates.
- **Note:** These results were from the pre-two-phase pipeline. The two-phase retrieval (diagnose → retrieve therapy) has not been re-evaluated yet.

## API Keys

Loaded from `.env` via `dotenv`: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `LLAMA_CLOUD_API_KEY`. Endpoint is `https://api.truhn.ai`.

### mwe.py — Standalone Test Script

Minimal working example for verifying Azure OpenAI connectivity. Uses `OPENAI_API_KEY` (not `AZURE_OPENAI_API_KEY`). Not part of the main application.

## Generated Artifacts (not in git)

- `ild_index/` — persisted vector store
- `parsed_jsons/` — LlamaParse output
- `__pycache__/`

**Warning:** There is no `.gitignore`. Be careful not to commit generated directories (`ild_index/`, `parsed_jsons/`, `__pycache__/`) or `.env`.
