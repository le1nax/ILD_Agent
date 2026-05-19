/*
 * citationMatcher.js — port of cliniva's PDF citation highlighting matcher.
 *
 * Exposes window.CitationMatcher with three functions:
 *   buildPageIndex(textContent, viewport) → pageIndex
 *   findMatches(query, pageIndex)         → [{origStart, origEnd}, ...]
 *   rangesToOverlays(ranges, pageIndex,
 *                    scale, viewportHeight) → [{left, top, width, height}, ...]
 *
 * Trimmed from cliniva/frontend/src/hooks/usePDFCitationSearch.ts:
 * Passes 1, 1b, 1c only (drop spatial-cluster set cover, pipe split,
 * cross-page, constraintRects).
 */
(function (global) {
  "use strict";

  // ── Normalization ─────────────────────────────────────────────────────────

  // Map all common dash variants to ASCII hyphen-minus, then NFC.
  function normalizeChars(text) {
    return text
      .replace(/[­‐‑‒–—―﹘﹣－]/g, "-")
      .normalize("NFC");
  }

  function isWordChar(ch) {
    return ch !== undefined && /\w/.test(ch);
  }

  // Returns true when the match at `index` with length `len` sits on word
  // boundaries within `text`. Prevents "stabil" matching inside "stabilität".
  // Multi-word queries skip the trailing-boundary check (PDF tables often
  // concatenate cells without spaces). A preceding digit is tolerated for
  // multi-word queries (footnote markers).
  function isWholeWordMatch(text, index, len, query) {
    const prevCh = text[index - 1];
    if (prevCh !== undefined && isWordChar(prevCh)) {
      if (!query.includes(" ") || /[a-zA-ZÀ-ɏ]/.test(prevCh)) {
        return false;
      }
    }
    if (query.includes(" ")) return true;
    if (isWordChar(text[index + len])) return false;
    return true;
  }

  // ── Page index construction ───────────────────────────────────────────────

  // Walks textContent.items (in content-stream order), builds:
  //   fullText             — raw concatenation with explicit gap/hyphen rules
  //   normalizedFullText   — lowercased, whitespace-collapsed, trimmed
  //   positionMapping      — normalizedIdx → raw fullText idx
  //   items                — [{str, transform, width, height, startPos, endPos}]
  function buildPageIndex(textContent, viewport) {
    let fullText = "";
    const items = [];

    textContent.items.forEach(function (item, index) {
      const itemText = normalizeChars(item.str || "");
      if (index > 0) {
        const prevItem = textContent.items[index - 1];
        const currentY = item.transform[5];
        const prevY = prevItem.transform[5];
        const currentX = item.transform[4];
        const prevX = prevItem.transform[4] + (prevItem.width || 0);
        const prevText = normalizeChars(prevItem.str || "");
        const prevEndsWithHyphen = prevText.endsWith("-");
        const currentStartsWithLetter = /^[a-zA-ZÀ-ɏ]/.test(itemText);
        const yDiff = Math.abs(currentY - prevY);
        const xGap = currentX - prevX;

        if (prevEndsWithHyphen && currentStartsWithLetter && yDiff > 2) {
          // Join hyphenated line-break: drop the trailing "-".
          fullText = fullText.slice(0, -1);
        } else if (yDiff > 2 || xGap > 5) {
          fullText += " ";
        }
      }
      const startPos = fullText.length;
      fullText += itemText;
      items.push({
        str: itemText,
        transform: item.transform || [0, 0, 0, 0, 0, 0],
        width: item.width || 0,
        height: item.height || 0,
        startPos: startPos,
        endPos: fullText.length,
      });
    });

    const normalizedFullText = fullText.replace(/\s+/g, " ").trim().toLowerCase();

    // positionMapping[normalizedIdx] = fullTextIdx
    // The normalized text has consecutive whitespace collapsed to one space
    // and is fully lowercased; rebuild a position-by-position map.
    const positionMapping = [];
    {
      let originalIndex = 0;
      let normalizedIndex = 0;
      const lowerFullText = fullText.toLowerCase();
      while (
        originalIndex < fullText.length &&
        normalizedIndex < normalizedFullText.length
      ) {
        const originalChar = lowerFullText[originalIndex];
        const normalizedChar = normalizedFullText[normalizedIndex];
        if (originalChar === normalizedChar) {
          positionMapping[normalizedIndex] = originalIndex;
          originalIndex++;
          normalizedIndex++;
        } else if (/\s/.test(originalChar)) {
          originalIndex++;
        } else {
          positionMapping[normalizedIndex] = originalIndex;
          originalIndex++;
          normalizedIndex++;
        }
      }
    }

    return {
      fullText: fullText,
      normalizedFullText: normalizedFullText,
      positionMapping: positionMapping,
      items: items,
      viewportHeight: viewport.height,
    };
  }

  // ── Matching ──────────────────────────────────────────────────────────────

  // Returns a single {origStart, origEnd} or null for one normalized-text match.
  function _normRangeToRaw(pageIndex, normStart, normEnd, fallbackLen) {
    const m = pageIndex.positionMapping;
    const origStart = m[normStart] !== undefined ? m[normStart] : normStart;
    const origEnd =
      m[normEnd - 1] !== undefined ? m[normEnd - 1] + 1 : origStart + fallbackLen;
    return { origStart: origStart, origEnd: origEnd };
  }

  // Pass 1 + 1b: substring search (optionally on a dehyphenated query).
  function _substringPass(pageIndex, normalizedQuery, fallbackLen) {
    const ranges = [];
    const norm = pageIndex.normalizedFullText;
    let idx = 0;
    while ((idx = norm.indexOf(normalizedQuery, idx)) !== -1) {
      if (!isWholeWordMatch(norm, idx, normalizedQuery.length, normalizedQuery)) {
        idx += 1;
        continue;
      }
      const end = idx + normalizedQuery.length;
      ranges.push(_normRangeToRaw(pageIndex, idx, end, fallbackLen));
      idx += 1;
    }
    return ranges;
  }

  // Pass 1c: anchor-based fuzzy match. Returns first hit only.
  function _anchorPass(pageIndex, normalizedQuery, fallbackLen) {
    const words = normalizedQuery.split(/\s+/);
    if (words.length < 4) return [];
    const anchorLen = Math.min(2, Math.floor(words.length / 3));
    const startAnchor = words.slice(0, anchorLen).join(" ");
    const endAnchor = words
      .slice(-anchorLen)
      .join(" ")
      .replace(/[.,;:!?)]+$/, "");
    if (!startAnchor || !endAnchor) return [];

    const norm = pageIndex.normalizedFullText;
    let aIdx = 0;
    while ((aIdx = norm.indexOf(startAnchor, aIdx)) !== -1) {
      if (!isWholeWordMatch(norm, aIdx, startAnchor.length, startAnchor)) {
        aIdx += 1;
        continue;
      }
      const searchFrom = aIdx + startAnchor.length;
      const searchUntil = Math.min(
        norm.length,
        searchFrom + normalizedQuery.length * 2
      );
      const eIdx = norm.indexOf(endAnchor, searchFrom);
      if (eIdx !== -1 && eIdx < searchUntil) {
        const matchEnd = eIdx + endAnchor.length;
        return [_normRangeToRaw(pageIndex, aIdx, matchEnd, fallbackLen)];
      }
      aIdx += 1;
    }
    return [];
  }

  // Run the 4-pass matcher (without Pass 2/3). Returns [{origStart, origEnd}].
  function findMatches(query, pageIndex) {
    if (!query || !query.trim()) return [];
    const trimmed = query.trim();
    const normalizedQuery = normalizeChars(trimmed)
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
    if (!normalizedQuery) return [];

    // Pass 1: exact normalized substring.
    let ranges = _substringPass(pageIndex, normalizedQuery, trimmed.length);
    if (ranges.length) return ranges;

    // Pass 1b: retry with hyphens between word chars removed.
    const dehyphenated = normalizedQuery.replace(/(\w)-\s*(\w)/g, "$1$2");
    if (dehyphenated !== normalizedQuery) {
      ranges = _substringPass(pageIndex, dehyphenated, trimmed.length);
      if (ranges.length) return ranges;
    }

    // Pass 1c: anchor-based fuzzy match (≥4-word queries).
    ranges = _anchorPass(pageIndex, normalizedQuery, trimmed.length);
    return ranges;
  }

  // ── Overlay computation ───────────────────────────────────────────────────

  // Compute overlay rect for a single text item given the char offsets within
  // it. Mirrors cliniva's createOverlayForItem. Horizontal-text fast path,
  // rotated-text 4-corner bbox fallback.
  function createOverlayForItem(item, charsBefore, numChars, viewportHeight, scale) {
    const t = item.transform;
    const a = t[0], b = t[1], e = t[4], f = t[5];
    const angleRad = Math.atan2(b, a);
    const rotated = Math.abs(angleRad) > 0.01;

    if (!rotated && item.width > 0) {
      const charWidth = item.width / (item.str.length || 1);
      const offsetX = charsBefore * charWidth;
      const hlWidth = Math.max(numChars * charWidth, 10);
      const itemHeight = item.height || Math.abs(t[3]) || 10;
      return {
        left: (e + offsetX) * scale,
        top: (viewportHeight - f - itemHeight) * scale,
        width: Math.max(hlWidth * scale * 1.025, 10),
        height: Math.max(itemHeight * scale * 1.025, 8),
      };
    }

    // Rotated or zero-width text: bounding box from 4 corners.
    const fontSize = Math.sqrt(a * a + b * b) || Math.abs(t[3]) || 10;
    const textAdvance =
      Math.sqrt(item.width * item.width + item.height * item.height) || fontSize;
    const charWidth = textAdvance / (item.str.length || 1);
    const offset = charsBefore * charWidth;
    const hlLen = Math.max(numChars * charWidth, 10);
    const cos = rotated ? Math.cos(angleRad) : 0;
    const sin = rotated ? Math.sin(angleRad) : 1;
    const sx = e + offset * cos;
    const sy = f + offset * sin;
    const corners = [
      [sx, sy],
      [sx + hlLen * cos, sy + hlLen * sin],
      [sx - fontSize * sin, sy + fontSize * cos],
      [sx + hlLen * cos - fontSize * sin, sy + hlLen * sin + fontSize * cos],
    ];
    const vpCorners = corners.map(function (c) {
      return [c[0] * scale, (viewportHeight - c[1]) * scale];
    });
    const xs = vpCorners.map(function (c) { return c[0]; });
    const ys = vpCorners.map(function (c) { return c[1]; });
    return {
      left: Math.min.apply(null, xs),
      top: Math.min.apply(null, ys),
      width: Math.max(Math.max.apply(null, xs) - Math.min.apply(null, xs), 10),
      height: Math.max(Math.max.apply(null, ys) - Math.min.apply(null, ys), 8),
    };
  }

  // Merge per-item rects on the same visual line into one bounding rect per
  // line. Inter-word gaps within a line are intentionally swallowed so
  // highlights don't look striped.
  function mergeOverlays(overlays) {
    if (overlays.length <= 1) return overlays.slice();
    const sorted = overlays.slice().sort(function (x, y) {
      return x.top - y.top || x.left - y.left;
    });
    const lines = [];
    let cur = [sorted[0]];
    for (let i = 1; i < sorted.length; i++) {
      const prev = cur[0];
      const next = sorted[i];
      if (Math.abs(next.top - prev.top) < prev.height * 0.8) {
        cur.push(next);
      } else {
        lines.push(cur);
        cur = [next];
      }
    }
    lines.push(cur);
    return lines.map(function (line) {
      let minL = Infinity, minT = Infinity, maxR = -Infinity, maxB = -Infinity;
      for (let j = 0; j < line.length; j++) {
        const o = line[j];
        if (o.left < minL) minL = o.left;
        if (o.top < minT) minT = o.top;
        if (o.left + o.width > maxR) maxR = o.left + o.width;
        if (o.top + o.height > maxB) maxB = o.top + o.height;
      }
      return { left: minL, top: minT, width: maxR - minL, height: maxB - minT };
    });
  }

  // Convert a list of {origStart, origEnd} ranges into merged CSS-pixel rects.
  function rangesToOverlays(ranges, pageIndex, scale, viewportHeight) {
    if (!ranges || !ranges.length) return [];
    const items = pageIndex.items;
    const raw = [];
    for (let r = 0; r < ranges.length; r++) {
      const range = ranges[r];
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.startPos >= range.origEnd) break; // items are in stream order
        if (item.endPos <= range.origStart) continue;
        const overlapStart = Math.max(range.origStart, item.startPos);
        const overlapEnd = Math.min(range.origEnd, item.endPos);
        if (overlapStart >= overlapEnd) continue;
        const charsBefore = overlapStart - item.startPos;
        const numChars = overlapEnd - overlapStart;
        raw.push(
          createOverlayForItem(item, charsBefore, numChars, viewportHeight, scale)
        );
      }
    }
    return mergeOverlays(raw);
  }

  global.CitationMatcher = {
    buildPageIndex: buildPageIndex,
    findMatches: findMatches,
    rangesToOverlays: rangesToOverlays,
  };
})(window);
