import json
import os
import uuid
import functools
import requests as http_requests
from flask import (
    Flask, render_template, request, Response, stream_with_context,
    send_from_directory, abort, session, redirect, url_for,
)
from graph import (
    State, retrieve_node, stream_retrieve_node, stream_generate,
    _build_provenance_details, PDF_DIR, REWRITE_WORD_THRESHOLD,
)

app = Flask(__name__)

# Disable browser caching for static files (JS, CSS) during development
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-me")


def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

USAGE_API_URL = "http://137.226.23.79:42161/usage/user/me"

# In-memory session storage: {session_id: [{"role": ..., "content": ...}, ...]}
_sessions: dict[str, list[dict]] = {}


def _get_api_usage():
    """Fetch current cumulative usage from the proxy API."""
    try:
        token = os.getenv("AZURE_OPENAI_API_KEY")
        r = http_requests.get(
            USAGE_API_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == os.getenv("APP_PASSWORD", "changeme"):
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Falsches Passwort."
    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
@login_required
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return {"error": "No question provided"}, 400

    # Resolve or create session
    session_id = data.get("session_id")
    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = []

    history = _sessions[session_id]

    def generate():
        state = State(question=question, history=list(history))

        # Snapshot usage before the query
        usage_before = _get_api_usage()

        # Send session_id to the client (so it can send it back on follow-ups)
        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"

        # Retrieve docs (streaming: yields per-query status events)
        for event_type, content in stream_retrieve_node(state):
            if event_type == "status":
                yield f"event: status\ndata: {json.dumps({'text': content})}\n\n"

        # Step: Query Decomposition
        diag_queries = state.get("diagnostic_queries", [])
        if len(diag_queries) > 1:
            yield f"event: step\ndata: {json.dumps({'title': 'Query Decomposition', 'summary': f'Generated {len(diag_queries)} diagnostic queries', 'details': chr(10).join(f'- {q}' for q in diag_queries)})}\n\n"

        # Step: Document Retrieval — with per-query provenance breakdown
        n_docs = len(state["docs"])
        provenance = state.get("retrieval_provenance", {})
        node_ids = state.get("_node_ids", [])
        if len(diag_queries) == 1:
            method = "hybrid (embedding + BM25) + ColBERT reranking"
            retrieval_details = f"**Method:** {method}\n**Total chunks:** {n_docs}"
        else:
            method = "hybrid (embedding + BM25) + RRF merging"
            retrieval_details = (
                f"**Method:** {method}\n**Total chunks:** {n_docs}\n\n"
                + _build_provenance_details(
                    provenance, state["docs"], diag_queries,
                    is_docs=True, node_ids=node_ids,
                )
            )
        yield f"event: step\ndata: {json.dumps({'title': 'Document Retrieval', 'summary': f'Retrieved {n_docs} chunks via hybrid search', 'details': retrieval_details})}\n\n"

        # Send sources
        sources = []
        for doc in state["docs"]:
            sources.append({
                "document": doc.metadata.get("document", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "text": doc.page_content,
            })
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        # Stream answer tokens (+ diagnose/therapy phases for complex cases)
        full_answer = ""
        for event_type, content in stream_generate(state):
            if event_type == "token":
                full_answer += content
                yield f"event: token\ndata: {json.dumps({'token': content})}\n\n"
            elif event_type == "replace":
                full_answer = content
                yield f"event: replace\ndata: {json.dumps({'answer': content})}\n\n"
            elif event_type == "status":
                yield f"event: status\ndata: {json.dumps({'text': content})}\n\n"
            elif event_type == "sources_update":
                yield f"event: sources_update\ndata: {json.dumps(content)}\n\n"
            elif event_type == "sources_replace":
                yield f"event: sources_replace\ndata: {json.dumps(content)}\n\n"
            elif event_type == "step":
                yield f"event: step\ndata: {json.dumps(content)}\n\n"
            elif event_type == "highlights":
                yield f"event: highlights\ndata: {json.dumps(content)}\n\n"

        # Persist conversation turn
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})

        # Snapshot usage after the query
        usage_after = _get_api_usage()

        # Build usage payload: estimated + actual
        estimated = state.get("usage", {})
        actual = {}
        if usage_before and usage_after:
            actual = {
                "input_tokens": usage_after["tk_in"] - usage_before["tk_in"],
                "output_tokens": usage_after["tk_out"] - usage_before["tk_out"],
                "total_tokens": usage_after["tk_total"] - usage_before["tk_total"],
                "cost_eur": round(float(usage_after["cost_eur"]) - float(usage_before["cost_eur"]), 6),
            }

        yield f"event: usage\ndata: {json.dumps({'estimated': estimated, 'actual': actual})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/pdf/<path:filename>")
@login_required
def serve_pdf(filename):
    """Serve a PDF from the guidelines directory."""
    if not filename.endswith(".pdf"):
        abort(404)
    pdf_dir = os.path.abspath(PDF_DIR)
    return send_from_directory(pdf_dir, filename, mimetype="application/pdf")


@app.route("/sessions/<session_id>", methods=["GET"])
@login_required
def get_session(session_id):
    if session_id not in _sessions:
        return {"error": "Session not found"}, 404
    return {"session_id": session_id, "history": _sessions[session_id]}


@app.route("/new_chat", methods=["POST"])
@login_required
def new_chat():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return {"ok": True}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
