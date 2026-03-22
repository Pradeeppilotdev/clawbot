from __future__ import annotations

import json
import re
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ToolHandler = Callable[[str], str]
WORKSPACE_ROOT = Path.cwd().resolve()


@dataclass
class Tool:
    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def call(self, name: str, tool_input: str) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Tool '{name}' is not available."
        try:
            return tool.handler(tool_input)
        except Exception as exc:  # pragma: no cover
            return f"Tool '{name}' failed: {exc}"


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="now",
            description="Returns current UTC time.",
            handler=lambda _: datetime.now(timezone.utc).isoformat(),
        )
    )

    registry.register(
        Tool(
            name="summarize",
            description="Summarizes input text into 3 concise bullet points.",
            handler=_summarize,
        )
    )

    registry.register(
        Tool(
            name="web_search",
            description="Searches the web and returns top results as JSON with title, url, snippet.",
            handler=_web_search,
        )
    )

    registry.register(
        Tool(
            name="fetch_url",
            description="Fetches a URL and returns JSON with title, url, and cleaned text excerpt.",
            handler=_fetch_url,
        )
    )

    registry.register(
        Tool(
            name="read_text_file",
            description="Reads a workspace text file and returns a clipped excerpt as JSON.",
            handler=_read_text_file,
        )
    )

    registry.register(
        Tool(
            name="csv_profile",
            description="Profiles a CSV file in the workspace and returns schema/stat summary JSON.",
            handler=_csv_profile,
        )
    )

    return registry


def _summarize(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "- No text supplied"

    chunks = [cleaned[i : i + 140] for i in range(0, min(len(cleaned), 420), 140)]
    bullets = [f"- {chunk}" for chunk in chunks[:3]]
    return "\n".join(bullets)


def _web_search(query: str) -> str:
    query = query.strip()
    if not query:
        return json.dumps({"query": "", "results": []}, ensure_ascii=True)

    # Primary path: RSS is more stable than parsing changing HTML markup.
    rss_xml = _http_get(
        "https://www.bing.com/search?format=rss"
        f"&setlang=en-US&cc=US&mkt=en-US&ensearch=1&q={quote_plus(query)}"
    )
    rss_results = _parse_bing_rss(rss_xml)
    rss_results = _rank_and_filter_results(query, rss_results)
    if rss_results:
        return json.dumps({"query": query, "results": rss_results[:5]}, ensure_ascii=True)

    html = _http_get(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}")
    if not html:
        return json.dumps(
            {"query": query, "results": [], "error": "Unable to fetch search results."},
            ensure_ascii=True,
        )

    link_re = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_re = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    )

    links = link_re.findall(html)
    snippets = snippet_re.findall(html)
    results: list[dict[str, str]] = []

    for idx, (href, raw_title) in enumerate(links[:5]):
        title = _clean_html(raw_title)
        url = _resolve_ddg_redirect(href)
        snippet_raw = ""
        if idx < len(snippets):
            snippet_raw = snippets[idx][0] or snippets[idx][1]
        snippet = _clean_html(snippet_raw)
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})

    return json.dumps({"query": query, "results": results}, ensure_ascii=True)


def _fetch_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return json.dumps(
            {"url": url, "error": "Only http/https URLs are allowed."},
            ensure_ascii=True,
        )

    html = _http_get(url)
    if not html:
        return json.dumps({"url": url, "error": "Failed to fetch URL."}, ensure_ascii=True)

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = _clean_html(title_match.group(1)) if title_match else ""
    text = _clean_html(html)

    return json.dumps(
        {
            "title": title,
            "url": url,
            "excerpt": text[:2200],
        },
        ensure_ascii=True,
    )


def _read_text_file(tool_input: str) -> str:
    path = _resolve_workspace_path(tool_input)
    if path is None:
        return json.dumps({"error": "Invalid path. Use a workspace-relative file path."}, ensure_ascii=True)
    if not path.exists() or not path.is_file():
        return json.dumps({"error": "File not found.", "path": str(path)}, ensure_ascii=True)

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return json.dumps({"error": f"Unable to read file: {exc}"}, ensure_ascii=True)

    clip = raw[:8000]
    return json.dumps(
        {
            "path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "chars": len(raw),
            "truncated": len(raw) > len(clip),
            "excerpt": clip,
        },
        ensure_ascii=True,
    )


def _csv_profile(tool_input: str) -> str:
    path = _resolve_workspace_path(tool_input)
    if path is None:
        return json.dumps({"error": "Invalid path. Use a workspace-relative CSV path."}, ensure_ascii=True)
    if not path.exists() or not path.is_file():
        return json.dumps({"error": "File not found.", "path": str(path)}, ensure_ascii=True)

    if path.suffix.lower() != ".csv":
        return json.dumps({"error": "Only .csv files are supported.", "path": str(path)}, ensure_ascii=True)

    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            rows: list[dict[str, str]] = []
            for idx, row in enumerate(reader):
                if idx >= 3000:
                    break
                rows.append({k: (v or "") for k, v in row.items()})
    except Exception as exc:
        return json.dumps({"error": f"Failed to parse CSV: {exc}"}, ensure_ascii=True)

    columns = reader.fieldnames or []
    profile: dict[str, dict[str, object]] = {}
    for col in columns:
        values = [r.get(col, "") for r in rows]
        non_empty = [v for v in values if str(v).strip()]
        numeric_values: list[float] = []
        for v in non_empty:
            try:
                numeric_values.append(float(str(v).replace(",", "")))
            except Exception:
                continue

        summary: dict[str, object] = {
            "non_empty": len(non_empty),
            "empty": len(values) - len(non_empty),
            "unique_sample": len(set(non_empty[:1000])),
        }
        if numeric_values and len(numeric_values) >= max(3, len(non_empty) // 2):
            summary["numeric"] = True
            summary["min"] = min(numeric_values)
            summary["max"] = max(numeric_values)
            summary["avg"] = round(sum(numeric_values) / len(numeric_values), 4)
        else:
            summary["numeric"] = False
        profile[col] = summary

    return json.dumps(
        {
            "path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "rows_sampled": len(rows),
            "columns": columns,
            "profile": profile,
        },
        ensure_ascii=True,
    )


def _http_get(url: str) -> str:
    req = Request(url, headers={"User-Agent": "clawbot/1.0 (+https://example.local)"})
    try:
        with urlopen(req, timeout=12) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _clean_html(value: str) -> str:
    value = re.sub(r"<script[\\s\\S]*?</script>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<style[\\s\\S]*?</style>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = unescape(value)
    return " ".join(value.split())


def _resolve_ddg_redirect(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return url


def _parse_bing_rss(xml_text: str) -> list[dict[str, str]]:
    if not xml_text.strip():
        return []

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []

    items = root.findall("./channel/item")
    results: list[dict[str, str]] = []

    for item in items:
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        snippet = (item.findtext("description") or "").strip()
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results


def _resolve_workspace_path(tool_input: str) -> Path | None:
    raw = tool_input.strip().strip('"').strip("'")
    if not raw:
        return None

    p = Path(raw)
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p

    try:
        resolved = p.resolve()
    except Exception:
        return None

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return None
    return resolved


def _rank_and_filter_results(query: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
    if not results:
        return []

    q = query.lower()
    required_keywords = ["hackathon", "judg", "agent", "rubric", "criteria", "devpost", "lablab"]
    blocked_domains = {"zhihu.com", "baidu.com"}

    ranked: list[tuple[int, dict[str, str]]] = []
    for item in results:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        domain = urlparse(url).netloc.lower()
        text = f"{title} {snippet}".lower()

        score = 0
        if any(k in text for k in required_keywords):
            score += 3
        if "hackathon" in q and "hackathon" in text:
            score += 3
        if ("judging" in q or "criteria" in q) and ("judg" in text or "criteria" in text):
            score += 3
        if "devpost" in domain or "lablab" in domain:
            score += 3
        if any(b in domain for b in blocked_domains):
            score -= 6

        if score >= 2:
            ranked.append((score, item))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked]
