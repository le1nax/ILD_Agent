import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.worker.min.mjs";

console.log("[app.js] Module loaded successfully");

// ── Highlight debug log (collected in memory, downloadable) ──
const _hlLog = [];
function hlLog(msg) {
  const ts = new Date().toISOString().slice(11, 23);
  _hlLog.push(`[${ts}] ${msg}`);
}
window._downloadHlLog = function() {
  const blob = new Blob([_hlLog.join("\n")], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "highlight-debug.txt";
  a.click();
};
console.log("[app.js] Highlight debug ready — call _downloadHlLog() in console to save log");

// ── Multi-chat state ──
let chats = [];           // [{id, sessionId, title, createdAt, pdfVisible}]
let activeChatId = null;
let currentSources = [];  // sources for the active chat's latest query
let currentHighlights = null;
const activeStreams = new Map(); // chatId -> { abortController }
const chatContainers = new Map(); // chatId -> detached DOM container for background chats
let chatDirty = false;    // whether active chat was modified in this tab

// ── DOM refs ──
const chatEl = document.getElementById("chat");
const questionEl = document.getElementById("question");
const submitBtn = document.getElementById("submit-btn");
const spinner = document.getElementById("spinner");
const errorEl = document.getElementById("error");
const pdfViewer = document.getElementById("pdf-viewer");
const pdfTitle = document.getElementById("pdf-title");
const pdfPageInfo = document.getElementById("pdf-page-info");

// Cache loaded PDF documents so we don't re-fetch the same file
const pdfCache = {}; // filename -> PDFDocumentProxy

// Each rendered source section gets an id so [Source N] can scroll to it
let sourceContainers = [];

// Pending exact-quote highlights — stored so late-rendering pages can apply them
let pendingHighlights = null;

// ── Utility ──

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setLoading(on) {
  submitBtn.disabled = on;
  spinner.classList.toggle("visible", on);
  if (on) spinner.textContent = "Retrieving documents...";
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.add("visible");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ── localStorage persistence ──

function saveChats({ skipMerge = false } = {}) {
  try {
    if (!skipMerge) {
      // Merge with other tabs: read current state, integrate our chats, write back
      const stored = localStorage.getItem("ild-rag-chats");
      if (stored) {
        try {
          const otherChats = JSON.parse(stored);
          const localIds = new Set(chats.map(c => c.id));
          for (const other of otherChats) {
            if (!localIds.has(other.id)) {
              chats.push(other);
            }
          }
        } catch (e) { /* ignore parse errors */ }
      }
    }
    localStorage.setItem("ild-rag-chats", JSON.stringify(chats));
  } catch (e) {
    console.warn("localStorage write failed:", e);
  }
}

function saveCurrentChatState() {
  if (!activeChatId || !chatDirty) return;
  const chat = chats.find(c => c.id === activeChatId);
  if (!chat) return;

  try {
    localStorage.setItem(`ild-chat-html-${chat.id}`, chatEl.innerHTML);
    localStorage.setItem(`ild-chat-sources-${chat.id}`, JSON.stringify(currentSources));
    if (currentHighlights) {
      localStorage.setItem(`ild-chat-highlights-${chat.id}`, JSON.stringify(currentHighlights));
    }
  } catch (e) {
    console.warn("localStorage write failed:", e);
  }
}

function saveChatState(chatId, sources, highlights) {
  try {
    let html;
    if (chatId === activeChatId) {
      html = chatEl.innerHTML;
    } else {
      const container = chatContainers.get(chatId);
      html = container ? container.innerHTML : null;
    }
    if (html !== null) {
      localStorage.setItem(`ild-chat-html-${chatId}`, html);
    }
    localStorage.setItem(`ild-chat-sources-${chatId}`, JSON.stringify(sources || []));
    if (highlights) {
      localStorage.setItem(`ild-chat-highlights-${chatId}`, JSON.stringify(highlights));
    }
  } catch (e) {
    console.warn("saveChatState failed:", e);
  }
}

// ── Chat management ──

function createNewChat(savePrevious = true) {
  if (savePrevious) {
    saveCurrentChatState();
  }

  // Move current chat's DOM nodes to detached container (don't abort its stream)
  if (activeChatId) {
    const container = document.createElement('div');
    while (chatEl.firstChild) container.appendChild(chatEl.firstChild);
    chatContainers.set(activeChatId, container);
  }

  const chat = {
    id: generateId(),
    sessionId: null,
    title: "New Chat",
    createdAt: Date.now(),
    pdfVisible: true,
  };

  chats.unshift(chat);
  activeChatId = chat.id;
  chatDirty = false;

  // New chat has no active stream — always reset loading state
  setLoading(false);

  // chatEl is already empty from the move above
  currentSources = [];
  currentHighlights = null;
  errorEl.classList.remove("visible");

  // Reset PDF panel
  pdfViewer.innerHTML = '<div class="pdf-placeholder">Click a source citation to view the PDF</div>';
  pdfTitle.textContent = "No PDF loaded";
  pdfPageInfo.textContent = "";
  sourceContainers = [];
  pendingHighlights = null;

  // Restore PDF panel visibility
  const pdfPanel = document.getElementById("pdf-panel");
  pdfPanel.classList.remove("collapsed");
  updatePdfToggleBtn();

  // Update title
  document.getElementById("chat-title").textContent = "ILD RAG Assistant";

  // Persist
  saveChats();
  localStorage.setItem("ild-rag-active-chat", chat.id);
  renderChatList();

  questionEl.focus();
}

function switchToChat(chatId) {
  if (chatId === activeChatId) return;

  // Save current state (don't abort running streams — they continue in background)
  saveCurrentChatState();

  // Move current chat's DOM nodes to detached container
  if (activeChatId) {
    const container = document.createElement('div');
    while (chatEl.firstChild) container.appendChild(chatEl.firstChild);
    chatContainers.set(activeChatId, container);
  }

  // Switch
  activeChatId = chatId;
  chatDirty = false;
  const chat = chats.find(c => c.id === chatId);

  // Restore DOM: prefer in-memory container (preserves live stream elements), fall back to localStorage
  const targetContainer = chatContainers.get(chatId);
  if (targetContainer) {
    while (targetContainer.firstChild) chatEl.appendChild(targetContainer.firstChild);
    chatContainers.delete(chatId);
  } else {
    const savedHtml = localStorage.getItem(`ild-chat-html-${chatId}`) || "";
    chatEl.innerHTML = savedHtml;
  }

  // Restore sources
  const savedSources = localStorage.getItem(`ild-chat-sources-${chatId}`);
  currentSources = savedSources ? JSON.parse(savedSources) : [];

  // Restore highlights
  const savedHighlights = localStorage.getItem(`ild-chat-highlights-${chatId}`);
  currentHighlights = savedHighlights ? JSON.parse(savedHighlights) : null;

  // Re-render PDF. Set pendingHighlights BEFORE renderAllSources so the
  // first pages that finish rendering pick them up during their own drain
  // (renderAllSources internally clears pendingHighlights at the top, then
  // renderSourcePage reads it again after each page completes).
  if (currentSources.length > 0) {
    if (currentHighlights) {
      pendingHighlights = currentHighlights;
    }
    renderAllSources(currentSources);
    if (currentHighlights) {
      pendingHighlights = currentHighlights;
    }
  } else {
    pdfViewer.innerHTML = '<div class="pdf-placeholder">Click a source citation to view the PDF</div>';
    pdfTitle.textContent = "No PDF loaded";
    pdfPageInfo.textContent = "";
    sourceContainers = [];
    pendingHighlights = null;
  }

  // Show loading if this chat has an active stream
  if (activeStreams.has(chatId)) {
    setLoading(true);
  } else {
    setLoading(false);
  }

  // Update PDF panel visibility
  const pdfPanel = document.getElementById("pdf-panel");
  if (chat.pdfVisible === false) {
    pdfPanel.classList.add("collapsed");
  } else {
    pdfPanel.classList.remove("collapsed");
  }
  updatePdfToggleBtn();

  // Update sidebar
  renderChatList();

  // Update title
  document.getElementById("chat-title").textContent = chat.title || "ILD RAG Assistant";

  // Persist active chat
  localStorage.setItem("ild-rag-active-chat", chatId);

  // Reset error state
  errorEl.classList.remove("visible");
}

function deleteChat(chatId) {
  const idx = chats.findIndex(c => c.id === chatId);
  if (idx === -1) return;

  // Abort any running stream for this chat
  if (activeStreams.has(chatId)) {
    activeStreams.get(chatId).abortController.abort();
    activeStreams.delete(chatId);
  }
  chatContainers.delete(chatId);

  const chat = chats[idx];

  // Clean up server session
  if (chat.sessionId) {
    fetch("/new_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: chat.sessionId }),
    });
  }

  // Clean up localStorage
  localStorage.removeItem(`ild-chat-html-${chatId}`);
  localStorage.removeItem(`ild-chat-sources-${chatId}`);
  localStorage.removeItem(`ild-chat-highlights-${chatId}`);

  // Remove from array
  chats.splice(idx, 1);
  saveChats({ skipMerge: true });

  // If deleted chat was active, switch to another
  if (chatId === activeChatId) {
    if (chats.length === 0) {
      createNewChat(false);
    } else {
      const newActive = chats[Math.min(idx, chats.length - 1)];
      switchToChat(newActive.id);
    }
  } else {
    renderChatList();
  }
}

function renderChatList() {
  const chatList = document.getElementById("chat-list");
  chatList.innerHTML = "";

  for (const chat of chats) {
    const item = document.createElement("div");
    item.className = "chat-item" + (chat.id === activeChatId ? " active" : "");
    item.dataset.chatId = chat.id;

    const title = document.createElement("span");
    title.className = "chat-item-title";
    title.textContent = chat.title || "New Chat";
    item.appendChild(title);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "chat-item-delete";
    deleteBtn.textContent = "\u2715";
    deleteBtn.title = "Delete chat";
    item.appendChild(deleteBtn);

    chatList.appendChild(item);
  }
}

// ── Sidebar / PDF toggles ──

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  sidebar.classList.toggle("collapsed");
  try {
    localStorage.setItem("ild-rag-sidebar-visible", !sidebar.classList.contains("collapsed"));
  } catch (e) { /* ignore */ }
}

function togglePdfPanel() {
  const pdfPanel = document.getElementById("pdf-panel");
  const wasCollapsed = pdfPanel.classList.contains("collapsed");
  pdfPanel.classList.toggle("collapsed");
  const isNowOpen = wasCollapsed;

  updatePdfToggleBtn();

  // Update chat state
  const chat = chats.find(c => c.id === activeChatId);
  if (chat) {
    chat.pdfVisible = isNowOpen;
    saveChats();
  }

  // Re-render sources if opening and sources exist (fixes width issues).
  // Set pendingHighlights BEFORE renderAllSources clears it; renderSourcePage
  // re-reads it after each page renders.
  if (isNowOpen && currentSources.length > 0) {
    setTimeout(() => {
      if (currentHighlights) {
        pendingHighlights = currentHighlights;
      }
      renderAllSources(currentSources);
      if (currentHighlights) {
        pendingHighlights = currentHighlights;
      }
    }, 350); // wait for CSS transition
  }
}

function updatePdfToggleBtn() {
  const pdfPanel = document.getElementById("pdf-panel");
  const btn = document.getElementById("pdf-toggle-btn");
  btn.textContent = pdfPanel.classList.contains("collapsed") ? "Show PDF" : "Hide PDF";
}

// ── PDF rendering ──

async function getPdfDoc(filename) {
  if (pdfCache[filename]) return pdfCache[filename];
  const url = `/pdf/${encodeURIComponent(filename)}`;
  const doc = await pdfjsLib.getDocument(url).promise;
  pdfCache[filename] = doc;
  return doc;
}

/**
 * Render all cited sources as stacked PDF pages in the viewer.
 */
async function renderAllSources(sources) {
  console.log(`[renderAllSources] called with ${sources.length} sources`);
  pdfViewer.innerHTML = "";
  sourceContainers = [];
  pendingHighlights = null;

  const count = sources.length;
  pdfTitle.textContent = `${count} cited passage${count !== 1 ? "s" : ""}`;
  pdfPageInfo.textContent = "";

  for (let i = 0; i < sources.length; i++) {
    const src = sources[i];
    const filename = src.document + ".pdf";

    // Parse page range
    const pageStr = String(src.page);
    const pageNums = [];
    const rangeMatch = pageStr.match(/^(\d+)\s*-\s*(\d+)$/);
    if (rangeMatch) {
      const start = parseInt(rangeMatch[1], 10);
      const end = parseInt(rangeMatch[2], 10);
      for (let p = start; p <= end; p++) pageNums.push(p);
    } else {
      pageNums.push(parseInt(pageStr, 10) || 1);
    }

    // Section wrapper
    const section = document.createElement("div");
    section.className = "pdf-source-section";
    section.id = `pdf-source-${i}`;

    // Label
    const label = document.createElement("div");
    label.className = "pdf-source-label";
    label.textContent = `[Source ${i + 1}] ${src.document} — Page ${src.page}`;
    section.appendChild(label);

    // Chunk text viewer (debug)
    const chunkToggle = document.createElement("div");
    chunkToggle.className = "chunk-viewer-toggle";
    chunkToggle.textContent = "Show LLM input";
    section.appendChild(chunkToggle);

    const chunkBody = document.createElement("div");
    chunkBody.className = "chunk-viewer-body";
    chunkBody.textContent = src.text;
    section.appendChild(chunkBody);

    // No individual listener — delegation on pdfViewer handles clicks

    pdfViewer.appendChild(section);
    sourceContainers.push(section);

    // Render all pages in the range (async, fire-and-forget)
    for (const pageNum of pageNums) {
      renderSourcePage(section, filename, pageNum, src.text);
    }
  }
}

async function renderSourcePage(section, filename, pageNum, highlightText) {
  console.log(`[renderSourcePage] ${filename} page ${pageNum}, section id: ${section.id}`);
  try {
    const pdfDoc = await getPdfDoc(filename);
    pageNum = Math.max(1, Math.min(pageNum, pdfDoc.numPages));

    const page = await pdfDoc.getPage(pageNum);
    // Use fallback width if viewer is collapsed/hidden
    const rawWidth = pdfViewer.clientWidth;
    const viewerWidth = rawWidth > 50 ? rawWidth - 32 : 600;
    const scale = viewerWidth / page.getViewport({ scale: 1 }).width;
    const viewport = page.getViewport({ scale });

    const container = document.createElement("div");
    container.className = "pdf-page-container";
    container.style.width = viewport.width + "px";
    container.style.height = viewport.height + "px";

    // Canvas
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    container.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    await page.render({ canvasContext: ctx, viewport }).promise;

    // Text layer (transparent — only for text selection)
    const textContent = await page.getTextContent();
    const textLayerDiv = document.createElement("div");
    textLayerDiv.className = "pdf-text-layer";
    container.appendChild(textLayerDiv);
    renderTextLayer(textLayerDiv, textContent, viewport);

    // Highlight overlay layer (absolute-positioned divs over the canvas)
    const highlightLayer = document.createElement("div");
    highlightLayer.className = "pdf-highlight-layer";
    container.appendChild(highlightLayer);

    // Cache the page index + render params on the container as DOM-node
    // properties (not dataset — these hold non-string objects, and DOM
    // properties survive appendChild/removeChild for chat-switch detach).
    // _viewportHeight MUST be the UNSCALED PDF page height (user-space
    // points), because item.transform[5] from getTextContent() is in user
    // space and the overlay y-flip formula is
    //   top = (viewportHeight - f - itemHeight) * scale.
    container._pageIndex = window.CitationMatcher
      ? window.CitationMatcher.buildPageIndex(textContent, viewport)
      : null;
    container._viewportHeight = viewport.height / scale;
    container._scale = scale;
    container._highlightLayer = highlightLayer;

    section.appendChild(container);

    // If exact-quote highlights arrived while this page was still rendering,
    // apply them now
    if (pendingHighlights) {
      const idx = sourceContainers.indexOf(section);
      if (idx !== -1 && String(idx) in pendingHighlights) {
        rehighlightSource(section, pendingHighlights[String(idx)]);
      }
    }
  } catch (e) {
    console.error(`[renderSourcePage] FAILED ${filename} page ${pageNum}:`, e);
    const err = document.createElement("div");
    err.className = "pdf-source-error";
    err.textContent = `Could not load ${filename}: ${e.message}`;
    section.appendChild(err);
  }
}

function renderTextLayer(div, textContent, viewport) {
  for (const item of textContent.items) {
    if (!item.str) continue;
    const span = document.createElement("span");
    span.textContent = item.str;

    const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
    const fontHeight = Math.hypot(tx[2], tx[3]);

    span.style.left = tx[4] + "px";
    span.style.top = (tx[5] - fontHeight) + "px";
    span.style.fontSize = fontHeight + "px";

    if (item.width > 0) {
      const desiredWidth = item.width * viewport.scale;
      span.style.letterSpacing = "0px";
      span.dataset.desiredWidth = desiredWidth;
    }

    div.appendChild(span);
  }

  for (const span of div.querySelectorAll("span")) {
    const desired = parseFloat(span.dataset.desiredWidth);
    if (desired && span.offsetWidth > 0) {
      span.style.transform = `scaleX(${desired / span.offsetWidth})`;
    }
  }
}

// Highlight rendering uses the cliniva matcher port in citationMatcher.js.
// Per-page index + scale are cached on each .pdf-page-container as DOM
// properties (_pageIndex, _viewportHeight, _scale, _highlightLayer).
function rehighlightSource(sectionEl, claimTexts) {
  hlLog(`[rehighlightSource] section=${sectionEl.id} claims=${JSON.stringify(claimTexts).slice(0,300)}`);
  if (!window.CitationMatcher) {
    hlLog("[rehighlightSource] CitationMatcher not loaded");
    return;
  }
  const containers = sectionEl.querySelectorAll(".pdf-page-container");
  for (const container of containers) {
    // Defensive no-op: page hasn't finished rendering yet. The
    // pendingHighlights drain in renderSourcePage will retry after render.
    if (!container._pageIndex || !container._highlightLayer) continue;

    container._highlightLayer.innerHTML = "";

    for (const claim of claimTexts) {
      if (!claim || typeof claim !== "string") continue;
      const ranges = window.CitationMatcher.findMatches(claim, container._pageIndex);
      if (!ranges.length) {
        hlLog(`[rehighlightSource] NO MATCH for "${claim.slice(0, 60)}..."`);
        continue;
      }
      const overlays = window.CitationMatcher.rangesToOverlays(
        ranges, container._pageIndex, container._scale, container._viewportHeight
      );
      hlLog(`[rehighlightSource] ${overlays.length} overlay(s) for "${claim.slice(0, 60)}..."`);
      for (const o of overlays) {
        const div = document.createElement("div");
        div.className = "citation-highlight";
        div.style.left = `${o.left}px`;
        div.style.top = `${o.top}px`;
        div.style.width = `${o.width}px`;
        div.style.height = `${o.height}px`;
        container._highlightLayer.appendChild(div);
      }
    }
  }
}


/**
 * Render additional therapy sources appended after the initial diagnostic sources.
 */
async function renderAdditionalSources(newSources, startIdx) {
  console.log(`[renderAdditionalSources] called with ${newSources.length} sources, startIdx=${startIdx}`);
  for (let i = 0; i < newSources.length; i++) {
    const src = newSources[i];
    const filename = src.document + ".pdf";

    const pageStr = String(src.page);
    const pageNums = [];
    const rangeMatch = pageStr.match(/^(\d+)\s*-\s*(\d+)$/);
    if (rangeMatch) {
      const start = parseInt(rangeMatch[1], 10);
      const end = parseInt(rangeMatch[2], 10);
      for (let p = start; p <= end; p++) pageNums.push(p);
    } else {
      pageNums.push(parseInt(pageStr, 10) || 1);
    }

    const section = document.createElement("div");
    section.className = "pdf-source-section";
    section.id = `pdf-source-${startIdx + i}`;

    const label = document.createElement("div");
    label.className = "pdf-source-label";
    label.style.background = "#2d8659";
    label.textContent = `[Source ${startIdx + i + 1}] ${src.document} — Page ${src.page} (therapy)`;
    section.appendChild(label);

    const chunkToggle = document.createElement("div");
    chunkToggle.className = "chunk-viewer-toggle";
    chunkToggle.textContent = "Show LLM input";
    section.appendChild(chunkToggle);

    const chunkBody = document.createElement("div");
    chunkBody.className = "chunk-viewer-body";
    chunkBody.textContent = src.text;
    section.appendChild(chunkBody);

    // No individual listener — delegation on pdfViewer handles clicks

    pdfViewer.appendChild(section);
    sourceContainers.push(section);

    for (const pageNum of pageNums) {
      renderSourcePage(section, filename, pageNum, src.text);
    }
  }

  const count = sourceContainers.length;
  pdfTitle.textContent = `${count} cited passage${count !== 1 ? "s" : ""}`;
}

// ── Scroll the PDF viewer to a specific source section ──
function scrollToSource(idx) {
  const section = sourceContainers[idx];
  if (section) {
    section.scrollIntoView({ behavior: "smooth", block: "start" });
    section.classList.add("flash");
    setTimeout(() => section.classList.remove("flash"), 1200);
  }
}

// ── Make [Source N] refs in rendered answer clickable ──
function linkifySources(bubbleEl, sources) {
  let html = bubbleEl.innerHTML;
  // Make inline [Source N] refs clickable
  html = html.replace(/\[Source\s+(\d+)\]/g, (match, num) => {
    const idx = parseInt(num, 10) - 1;
    if (idx >= 0 && idx < sources.length) {
      return `<span class="source-ref" data-source-idx="${idx}">${match}</span>`;
    }
    return match;
  });

  // Also linkify source names in the "Sources:" section at the end.
  // Match lines like "Document Name, Page 46-47" and add a clickable [N] prefix.
  for (let i = 0; i < sources.length; i++) {
    const src = sources[i];
    // Escape special regex chars in document name
    const escapedName = src.document.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // Match the document name (possibly with minor HTML tags around it)
    const nameRe = new RegExp(`(?<!\\[Source \\d+\\]\\s*)${escapedName}`, "g");
    html = html.replace(nameRe, (match) => {
      return `<span class="source-ref" data-source-idx="${i}">[Source ${i + 1}]</span> ${match}`;
    });
  }

  bubbleEl.innerHTML = html;
  // No individual listeners — delegation on #chat handles clicks
}

// ── Chat DOM functions ──

function addUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg user";
  wrapper.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatEl.appendChild(wrapper);
  scrollToBottom();
}

function createAssistantMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "msg assistant";

  const sourcesToggle = document.createElement("div");
  sourcesToggle.className = "sources-toggle";
  sourcesToggle.style.display = "none";
  wrapper.appendChild(sourcesToggle);

  const sourcesBody = document.createElement("div");
  sourcesBody.className = "sources-body";
  sourcesBody.innerHTML = "<ul></ul>";
  wrapper.appendChild(sourcesBody);

  // No individual listener — delegation on #chat handles .sources-toggle clicks

  const stepsContainer = document.createElement("div");
  stepsContainer.className = "pipeline-steps";
  wrapper.appendChild(stepsContainer);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrapper.appendChild(bubble);

  const usageBar = document.createElement("div");
  usageBar.className = "usage-bar";
  wrapper.appendChild(usageBar);

  const downloadBtn = document.createElement("button");
  downloadBtn.className = "download-pdf-btn";
  downloadBtn.type = "button";
  downloadBtn.textContent = "Download PDF";
  downloadBtn.title = "Download this response as a PDF";
  wrapper.appendChild(downloadBtn);

  chatEl.appendChild(wrapper);
  scrollToBottom();

  return { wrapper, sourcesToggle, sourcesBody, stepsContainer, bubble, usageBar, downloadBtn };
}

// ── Helper: build source list item (no individual listeners) ──
function buildSourceLi(src, idx, isTherapy) {
  const li = document.createElement("li");
  li.dataset.sourceIdx = idx;

  const header = document.createElement("span");
  const color = isTherapy ? "#2d8659" : "#4361ee";
  header.innerHTML =
    `<span style="color:${color};font-weight:700;margin-right:0.3rem">[${idx + 1}]</span>` +
    `<span class="doc-name">${escapeHtml(src.document)}</span> &mdash; ` +
    `<span class="page-num">Page ${escapeHtml(String(src.page))}</span>` +
    (isTherapy ? ` <span style="color:#2d8659;font-size:0.75rem;font-style:italic">(therapy)</span>` : "");
  li.appendChild(header);

  if (src.text) {
    const toggle = document.createElement("span");
    toggle.className = "chunk-toggle";
    toggle.textContent = "[show chunk]";
    li.appendChild(toggle);

    const chunk = document.createElement("div");
    chunk.className = "chunk-text";
    chunk.textContent = src.text;
    li.appendChild(chunk);
  }

  return li;
}

// ── Main query ──

async function submitQuestion() {
  const question = questionEl.value.trim();
  if (!question) return;

  errorEl.classList.remove("visible");
  questionEl.value = "";

  addUserMessage(question);
  setLoading(true);
  chatDirty = true;

  // Capture chat context for this stream (survives chat switches)
  const streamChatId = activeChatId;
  const isActive = () => activeChatId === streamChatId;

  // Auto-title from first user message
  const chat = chats.find(c => c.id === streamChatId);
  if (chat && (!chat.title || chat.title === "New Chat")) {
    chat.title = question.length > 40 ? question.slice(0, 40) + "..." : question;
    saveChats();
    renderChatList();
    document.getElementById("chat-title").textContent = chat.title;
  }

  const msg = createAssistantMessage();
  let answer = "";
  let streamSources = [];
  let streamHighlights = null;

  // Abort any existing stream for THIS chat only
  if (activeStreams.has(streamChatId)) {
    activeStreams.get(streamChatId).abortController.abort();
    activeStreams.delete(streamChatId);
  }
  const abortController = new AbortController();
  activeStreams.set(streamChatId, { abortController });

  try {
    const chatSessionId = chat ? chat.sessionId : null;
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: chatSessionId }),
      signal: abortController.signal,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let eventType = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          eventType = line.slice(7);
        } else if (line.startsWith("data: ") && eventType) {
          const data = JSON.parse(line.slice(6));

          console.log(`[SSE] event: ${eventType} (active: ${isActive()})`);

          if (eventType === "session") {
            if (chat) {
              chat.sessionId = data.session_id;
              saveChats();
            }

          } else if (eventType === "sources") {
            streamSources = data;
            if (isActive()) currentSources = [...streamSources];
            try { localStorage.setItem(`ild-chat-sources-${streamChatId}`, JSON.stringify(streamSources)); } catch(e) {}

            const ul = msg.sourcesBody.querySelector("ul");
            for (let i = 0; i < data.length; i++) {
              ul.appendChild(buildSourceLi(data[i], i, false));
            }
            msg.sourcesToggle.textContent = `Show sources (${data.length})`;
            msg.sourcesToggle.style.display = "block";

            if (isActive()) {
              spinner.textContent = "Generating answer...";
              renderAllSources(streamSources);

              const pdfPanel = document.getElementById("pdf-panel");
              if (pdfPanel.classList.contains("collapsed")) {
                pdfPanel.classList.remove("collapsed");
                updatePdfToggleBtn();
                if (chat) {
                  chat.pdfVisible = true;
                  saveChats();
                }
              }
            }

          } else if (eventType === "status") {
            if (isActive()) spinner.textContent = data.text;

          } else if (eventType === "step") {
            const step = document.createElement("div");
            step.className = "step";

            const stepHeader = document.createElement("div");
            stepHeader.className = "step-header";
            stepHeader.innerHTML =
              `<span class="step-title">${escapeHtml(data.title)}</span>` +
              `<span class="step-summary">${escapeHtml(data.summary)}</span>` +
              `<span class="step-chevron">&#9656;</span>`;
            step.appendChild(stepHeader);

            const stepDetails = document.createElement("div");
            stepDetails.className = "step-details";
            stepDetails.innerHTML = marked.parse(data.details || "");
            step.appendChild(stepDetails);

            msg.stepsContainer.appendChild(step);
            if (isActive()) {
              spinner.textContent = data.title + "...";
              scrollToBottom();
            }

            if (data.title === "Preliminary Diagnosis" && chat) {
              const diagTitle = (data.summary || "").replace(/^[-\s*]+/, "").trim();
              if (diagTitle) {
                chat.title = diagTitle.length > 60 ? diagTitle.slice(0, 60) + "..." : diagTitle;
                saveChats();
                renderChatList();
                if (isActive()) {
                  document.getElementById("chat-title").textContent = chat.title;
                }
              }
            }

          } else if (eventType === "sources_replace") {
            streamSources = data;
            if (isActive()) currentSources = [...streamSources];
            try { localStorage.setItem(`ild-chat-sources-${streamChatId}`, JSON.stringify(streamSources)); } catch(e) {}

            const replaceUl = msg.sourcesBody.querySelector("ul");
            replaceUl.innerHTML = "";
            for (let i = 0; i < streamSources.length; i++) {
              replaceUl.appendChild(buildSourceLi(streamSources[i], i, false));
            }
            msg.sourcesToggle.textContent =
              (msg.sourcesBody.classList.contains("open") ? "Hide" : "Show") +
              ` sources (${streamSources.length})`;

            if (isActive()) renderAllSources(streamSources);

          } else if (eventType === "sources_update") {
            const newSources = data;
            const startIdx = streamSources.length;
            streamSources = streamSources.concat(newSources);
            if (isActive()) currentSources = [...streamSources];
            try { localStorage.setItem(`ild-chat-sources-${streamChatId}`, JSON.stringify(streamSources)); } catch(e) {}

            const ul = msg.sourcesBody.querySelector("ul");
            for (let i = 0; i < newSources.length; i++) {
              ul.appendChild(buildSourceLi(newSources[i], startIdx + i, true));
            }
            msg.sourcesToggle.textContent =
              (msg.sourcesBody.classList.contains("open") ? "Hide" : "Show") +
              ` sources (${streamSources.length})`;

            if (isActive()) renderAdditionalSources(newSources, startIdx);

          } else if (eventType === "token") {
            answer += data.token;
            msg.bubble.innerHTML = marked.parse(answer);
            if (isActive()) scrollToBottom();

          } else if (eventType === "replace") {
            answer = data.answer;
            msg.bubble.innerHTML = marked.parse(answer);
            if (isActive()) scrollToBottom();

          } else if (eventType === "highlights") {
            streamHighlights = data;
            console.log("[SSE highlights]", {
              keys: Object.keys(data || {}),
              counts: Object.fromEntries(Object.entries(data || {}).map(([k, v]) => [k, Array.isArray(v) ? v.length : 0])),
              sample: Object.values(data || {})[0],
            });
            try { localStorage.setItem(`ild-chat-highlights-${streamChatId}`, JSON.stringify(data)); } catch(e) {}
            if (isActive()) {
              pendingHighlights = data;
              currentHighlights = data;
              for (let i = 0; i < sourceContainers.length; i++) {
                if (String(i) in data) {
                  rehighlightSource(sourceContainers[i], data[String(i)]);
                }
              }
            }

          } else if (eventType === "usage") {
            const est = data.estimated || {};
            const act = data.actual || {};
            let html = "";

            if (est.total_tokens) {
              const estCost = est.cost_usd < 0.01
                ? `$${est.cost_usd.toFixed(6)}`
                : `$${est.cost_usd.toFixed(4)}`;
              html += `<span class="usage-label">Estimated:</span>` +
                `<span>${est.input_tokens} in / ${est.output_tokens} out</span>` +
                `<span class="cost">${estCost}</span>`;
            }

            if (act.total_tokens) {
              const actCost = `\u20AC${act.cost_eur.toFixed(6)}`;
              html += `<span class="usage-sep"></span>` +
                `<span class="usage-label">Actual:</span>` +
                `<span>${act.input_tokens} in / ${act.output_tokens} out</span>` +
                `<span class="cost">${actCost}</span>`;
            }

            if (html) {
              msg.usageBar.innerHTML = html;
              msg.usageBar.classList.add("visible");
            }

          } else if (eventType === "done") {
            if (isActive()) setLoading(false);
            linkifySources(msg.bubble, streamSources);
            for (const det of msg.stepsContainer.querySelectorAll(".step-details.open")) {
              det.classList.remove("open");
              det.parentElement.querySelector(".step-chevron").innerHTML = "&#9656;";
            }
            if (msg.downloadBtn) msg.downloadBtn.classList.add("visible");

            // Hide uncited sources in the PDF viewer and add toggle button
            hlLog(`[done] isActive=${isActive()} answerLen=${answer.length} containers=${sourceContainers.length} citations=${[...answer.matchAll(/\[Source\s+(\d+)\]/g)].map(m=>m[1]).join(",")}`);
            if (isActive() && answer && sourceContainers.length > 0) {
              const citedIndices = new Set();
              for (const m of answer.matchAll(/\[Source\s+(\d+)\]/g)) {
                citedIndices.add(parseInt(m[1], 10) - 1);
              }
              let uncitedCount = 0;
              for (let i = 0; i < sourceContainers.length; i++) {
                if (!citedIndices.has(i)) {
                  sourceContainers[i].classList.add("uncited-source");
                  sourceContainers[i].style.display = "none";
                  uncitedCount++;
                }
              }
              if (uncitedCount > 0) {
                // Add toggle button at the top of the PDF viewer
                const existingToggle = pdfViewer.querySelector(".uncited-toggle");
                if (existingToggle) existingToggle.remove();
                const toggleBtn = document.createElement("button");
                toggleBtn.className = "uncited-toggle";
                toggleBtn.textContent = `Show ${uncitedCount} unused sources`;
                toggleBtn.dataset.visible = "false";
                toggleBtn.addEventListener("click", () => {
                  const show = toggleBtn.dataset.visible === "false";
                  toggleBtn.dataset.visible = show ? "true" : "false";
                  toggleBtn.textContent = show
                    ? `Hide ${uncitedCount} unused sources`
                    : `Show ${uncitedCount} unused sources`;
                  for (const sec of pdfViewer.querySelectorAll(".uncited-source")) {
                    sec.style.display = show ? "" : "none";
                  }
                });
                pdfViewer.insertBefore(toggleBtn, pdfViewer.firstChild);
                // Update title to reflect cited count
                const citedCount = sourceContainers.length - uncitedCount;
                pdfTitle.textContent = `${citedCount} cited passage${citedCount !== 1 ? "s" : ""} (${uncitedCount} unused)`;
              }
            }

            saveChatState(streamChatId, streamSources, streamHighlights);
          }

          eventType = null;
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      if (isActive()) showError("Error: " + e.message);
    }
  } finally {
    if (isActive()) setLoading(false);
    activeStreams.delete(streamChatId);
    saveChatState(streamChatId, streamSources, streamHighlights);
  }
}

// ── Download PDF (client-side via html2pdf.js) ──
function _findPrecedingQuestion(assistantWrapper) {
  let prev = assistantWrapper.previousElementSibling;
  while (prev && !prev.classList.contains("user")) {
    prev = prev.previousElementSibling;
  }
  return prev ? (prev.querySelector(".bubble")?.innerText || "").trim() : "";
}

function _renderSourcesForPdf(sources) {
  if (!sources || !sources.length) return "";
  const items = sources.map((src, i) => {
    const doc = escapeHtml(src.document || "Unknown");
    const page = escapeHtml(String(src.page || "?"));
    const section = src.section ? ` — <em>${escapeHtml(src.section)}</em>` : "";
    return `<li style="margin-bottom:6px"><strong>[Source ${i + 1}]</strong> ${doc} — Page ${page}${section}</li>`;
  }).join("");
  return (
    '<h2 style="font-size:13pt;margin:18px 0 8px;color:#1a1a2e;border-bottom:1px solid #ccc;padding-bottom:3px">Cited Sources</h2>' +
    `<ol style="padding-left:24px;margin:0;font-size:10pt">${items}</ol>`
  );
}

async function downloadAnswerAsPdf(assistantWrapper, sources) {
  if (typeof html2pdf === "undefined") {
    showError("PDF library failed to load. Check your network and reload.");
    return;
  }
  const btn = assistantWrapper.querySelector(".download-pdf-btn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Generating PDF…";
  }

  let root = null;
  let coverOverlay = null;
  try {
    const bubble = assistantWrapper.querySelector(".bubble");
    if (!bubble) throw new Error("answer bubble not found");

    // Clone the answer bubble so we don't mutate the live DOM.
    const answerClone = bubble.cloneNode(true);
    // Strip the [Source N] hover interaction so links don't appear blue/dotted
    // in the PDF.
    answerClone.querySelectorAll(".source-ref").forEach((el) => {
      const span = document.createElement("span");
      span.textContent = el.textContent;
      span.style.fontWeight = "600";
      el.replaceWith(span);
    });
    answerClone.style.cssText =
      "background:#fff;padding:0;box-shadow:none;border-radius:0;" +
      "font-size:11pt;line-height:1.55;color:#1a1a2e;";

    const question = _findPrecedingQuestion(assistantWrapper);
    const stamp = new Date().toLocaleString();

    root = document.createElement("div");
    // 180mm content width inside an A4 (210mm) with 15mm side margins.
    // Sized in mm because html2pdf's default unit is mm and html2canvas
    // computes element bounds from layout (which works fine with mm).
    root.style.cssText =
      "width:180mm;padding:0;margin:0;" +
      "font-family:Arial,Helvetica,sans-serif;color:#1a1a2e;" +
      "font-size:11pt;line-height:1.55;background:#ffffff;" +
      "box-sizing:border-box;";
    root.innerHTML =
      '<div style="border-bottom:2px solid #4361ee;padding-bottom:6px;margin-bottom:14px">' +
        '<h1 style="font-size:18pt;margin:0;color:#1a1a2e;font-weight:700">ILD RAG Assistant</h1>' +
        `<div style="color:#666;font-size:9pt;margin-top:2px">${escapeHtml(stamp)}</div>` +
      "</div>" +
      (question
        ? '<h2 style="font-size:13pt;margin:0 0 6px;color:#1a1a2e;border-bottom:1px solid #ccc;padding-bottom:3px;font-weight:700">Question</h2>' +
          `<div style="margin-bottom:18px;white-space:pre-wrap">${escapeHtml(question)}</div>`
        : "") +
      '<h2 style="font-size:13pt;margin:0 0 6px;color:#1a1a2e;border-bottom:1px solid #ccc;padding-bottom:3px;font-weight:700">Answer</h2>' +
      '<div id="__pdf_answer__" style="margin-bottom:18px"></div>' +
      _renderSourcesForPdf(sources);
    root.querySelector("#__pdf_answer__").appendChild(answerClone);

    // Most reliable approach for html2canvas: attach the element visibly to
    // the body, then cover it with an opaque overlay. html2canvas reads
    // computed styles, so the element must be in normal flow with non-zero
    // dimensions. Tricks like position:fixed + z-index:-99999 are known to
    // cause blank canvases in some browser/version combos.
    root.style.position = "absolute";
    root.style.top = "0";
    root.style.left = "0";
    document.body.appendChild(root);

    // Cover the screen so the user doesn't see the briefly-attached element.
    coverOverlay = document.createElement("div");
    coverOverlay.style.cssText =
      "position:fixed;inset:0;background:rgba(245,247,250,0.98);" +
      "z-index:99999;display:flex;align-items:center;justify-content:center;" +
      "font-family:Arial,sans-serif;font-size:14px;color:#4361ee;";
    coverOverlay.textContent = "Generating PDF…";
    document.body.appendChild(coverOverlay);

    // Wait one paint cycle so layout is computed before html2canvas snapshots.
    await new Promise((r) => requestAnimationFrame(() => r()));

    const dateSlug = new Date().toISOString().slice(0, 10);
    const opt = {
      margin: [15, 15, 15, 15], // mm
      filename: `ild-response-${dateSlug}.pdf`,
      image: { type: "jpeg", quality: 0.95 },
      html2canvas: {
        scale: 2,
        useCORS: true,
        backgroundColor: "#ffffff",
        logging: false,
      },
      jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
      pagebreak: { mode: ["css", "legacy"] },
    };

    console.log("[downloadAnswerAsPdf] rendering root", {
      width: root.offsetWidth,
      height: root.offsetHeight,
      textPreview: root.innerText.slice(0, 200),
    });

    await html2pdf().set(opt).from(root).save();
  } catch (e) {
    console.error("[downloadAnswerAsPdf] failed:", e);
    showError("Could not generate PDF: " + e.message);
  } finally {
    if (root && root.parentNode) root.remove();
    if (coverOverlay && coverOverlay.parentNode) coverOverlay.remove();
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Download PDF";
    }
  }
}

// ── Event delegation on #chat ──
chatEl.addEventListener("click", (e) => {
  // 0. Download PDF button on an assistant message
  const dlBtn = e.target.closest(".download-pdf-btn");
  if (dlBtn) {
    const wrapper = dlBtn.closest(".msg.assistant");
    if (wrapper) downloadAnswerAsPdf(wrapper, currentSources || []);
    return;
  }

  // 1. Source reference in answer text (e.g., [Source 1])
  const sourceRef = e.target.closest(".source-ref");
  if (sourceRef) {
    e.preventDefault();
    const idx = parseInt(sourceRef.dataset.sourceIdx, 10);
    // Auto-open PDF panel if collapsed
    const pdfPanel = document.getElementById("pdf-panel");
    if (pdfPanel.classList.contains("collapsed")) {
      togglePdfPanel();
    }
    scrollToSource(idx);
    return;
  }

  // 2. Sources toggle (show/hide source list)
  const sourcesToggle = e.target.closest(".sources-toggle");
  if (sourcesToggle) {
    const wrapper = sourcesToggle.closest(".msg");
    const sourcesBody = wrapper.querySelector(".sources-body");
    sourcesBody.classList.toggle("open");
    const open = sourcesBody.classList.contains("open");
    const count = sourcesBody.querySelectorAll("li").length;
    sourcesToggle.textContent = (open ? "Hide" : "Show") + ` sources (${count})`;
    return;
  }

  // 3. Chunk toggle in source list (must come before source-li)
  const chunkToggle = e.target.closest(".chunk-toggle");
  if (chunkToggle) {
    const li = chunkToggle.closest("li");
    const chunk = li.querySelector(".chunk-text");
    if (chunk) {
      const open = chunk.classList.toggle("open");
      chunkToggle.textContent = open ? "[hide chunk]" : "[show chunk]";
    }
    return;
  }

  // 4. Source list item click (scroll to PDF source)
  const sourceLi = e.target.closest(".sources-body li");
  if (sourceLi) {
    const idx = parseInt(sourceLi.dataset.sourceIdx, 10);
    if (!isNaN(idx)) {
      const pdfPanel = document.getElementById("pdf-panel");
      if (pdfPanel.classList.contains("collapsed")) {
        togglePdfPanel();
      }
      scrollToSource(idx);
    }
    return;
  }

  // 5. Step header click (toggle step details)
  const stepHeader = e.target.closest(".step-header");
  if (stepHeader) {
    const step = stepHeader.closest(".step");
    const details = step.querySelector(".step-details");
    const open = details.classList.toggle("open");
    step.querySelector(".step-chevron").innerHTML = open ? "&#9662;" : "&#9656;";
    return;
  }
});

// ── Event delegation on sidebar chat list ──
document.getElementById("chat-list").addEventListener("click", (e) => {
  const deleteBtn = e.target.closest(".chat-item-delete");
  if (deleteBtn) {
    e.stopPropagation();
    const item = deleteBtn.closest(".chat-item");
    deleteChat(item.dataset.chatId);
    return;
  }

  const item = e.target.closest(".chat-item");
  if (item) {
    switchToChat(item.dataset.chatId);
  }
});

// ── Event delegation on PDF viewer (chunk-viewer-toggle) ──
pdfViewer.addEventListener("click", (e) => {
  const toggle = e.target.closest(".chunk-viewer-toggle");
  if (toggle) {
    const section = toggle.closest(".pdf-source-section");
    const body = section.querySelector(".chunk-viewer-body");
    if (body) {
      const open = body.classList.toggle("open");
      toggle.textContent = open ? "Hide LLM input" : "Show LLM input";
    }
  }
});

// ── Button listeners ──
submitBtn.addEventListener("click", submitQuestion);
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitQuestion();
  }
});

document.getElementById("new-chat-sidebar-btn").addEventListener("click", () => createNewChat());
document.getElementById("sidebar-toggle-btn").addEventListener("click", toggleSidebar);
document.getElementById("pdf-toggle-btn").addEventListener("click", togglePdfPanel);
document.getElementById("pdf-close-btn").addEventListener("click", togglePdfPanel);

// ── Init ──
function init() {
  // Load chats from localStorage
  const savedChats = localStorage.getItem("ild-rag-chats");
  if (savedChats) {
    try {
      chats = JSON.parse(savedChats);
    } catch (e) {
      chats = [];
    }
  }

  if (chats.length === 0) {
    createNewChat(false);
    return;
  }

  // Restore active chat
  const savedActive = localStorage.getItem("ild-rag-active-chat");
  const targetId = savedActive && chats.find(c => c.id === savedActive)
    ? savedActive
    : chats[0].id;
  activeChatId = targetId;

  // Restore chat HTML
  const savedHtml = localStorage.getItem(`ild-chat-html-${targetId}`);
  if (savedHtml) chatEl.innerHTML = savedHtml;

  // Restore sources
  const savedSources = localStorage.getItem(`ild-chat-sources-${targetId}`);
  currentSources = savedSources ? JSON.parse(savedSources) : [];

  // Restore highlights
  const savedHighlightsStr = localStorage.getItem(`ild-chat-highlights-${targetId}`);
  currentHighlights = savedHighlightsStr ? JSON.parse(savedHighlightsStr) : null;

  // Render PDF
  if (currentSources.length > 0) {
    renderAllSources(currentSources);
    // Set pendingHighlights after renderAllSources so late-rendering pages pick them up
    if (currentHighlights) {
      pendingHighlights = currentHighlights;
    }
  }

  // Update title
  const chat = chats.find(c => c.id === targetId);
  document.getElementById("chat-title").textContent = chat?.title || "ILD RAG Assistant";

  // Restore PDF visibility
  if (chat?.pdfVisible === false) {
    document.getElementById("pdf-panel").classList.add("collapsed");
  }
  updatePdfToggleBtn();

  // Restore sidebar state
  const sidebarVisible = localStorage.getItem("ild-rag-sidebar-visible");
  if (sidebarVisible === "false") {
    document.getElementById("sidebar").classList.add("collapsed");
  }

  renderChatList();
}

init();

// Save state when leaving page
window.addEventListener("beforeunload", () => {
  saveCurrentChatState();
  // Save background chat containers so their state persists across reload
  for (const [chatId, container] of chatContainers) {
    try {
      localStorage.setItem(`ild-chat-html-${chatId}`, container.innerHTML);
    } catch (e) { /* ignore */ }
  }
});

// Sync chat list across tabs: when another tab modifies localStorage, update sidebar
window.addEventListener("storage", (e) => {
  if (e.key !== "ild-rag-chats" || !e.newValue) return;

  try {
    const otherChats = JSON.parse(e.newValue);
    const localIds = new Set(chats.map(c => c.id));
    const otherIds = new Set(otherChats.map(c => c.id));

    // Add new chats from the other tab
    let changed = false;
    for (const other of otherChats) {
      if (!localIds.has(other.id)) {
        chats.push(other);
        changed = true;
      }
    }

    // Remove chats deleted by the other tab (but never remove the active chat)
    for (let i = chats.length - 1; i >= 0; i--) {
      if (!otherIds.has(chats[i].id) && chats[i].id !== activeChatId) {
        chats.splice(i, 1);
        changed = true;
      }
    }

    // Update titles/metadata from other tab for chats we share
    for (const other of otherChats) {
      if (other.id === activeChatId) continue; // don't overwrite our active chat
      const local = chats.find(c => c.id === other.id);
      if (local && local.title !== other.title) {
        local.title = other.title;
        changed = true;
      }
    }

    if (changed) {
      renderChatList();
    }
  } catch (err) {
    console.warn("storage sync failed:", err);
  }
});
