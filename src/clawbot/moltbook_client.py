from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


@dataclass
class MoltbookPost:
    id: str
    title: str
    content: str
    author: str
    submolt: str
    upvotes: int
    comment_count: int
    created_at: str


@dataclass
class MoltbookComment:
    id: str
    content: str
    author: str
    post_id: str
    upvotes: int
    created_at: str


class MoltbookClient:
    def __init__(self, api_key: str, base_url: str = "https://www.moltbook.com/api/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url

    def _request(self, method: str, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make authenticated HTTP request to Moltbook API."""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = json.dumps(data).encode("utf-8") if data else None
        request = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read().decode("utf-8", errors="ignore")
                return json.loads(raw)
        except HTTPError as exc:
            return {
                "error": f"HTTP {exc.code}",
                "message": exc.reason,
            }
        except URLError as exc:
            return {
                "error": "Network error",
                "message": str(exc),
            }

    def get_feed(self, limit: int = 20, sort: str = "hot") -> list[MoltbookPost]:
        """Fetch hot posts from the feed."""
        resp = self._request("GET", f"/feed?sort={sort}&limit={limit}")
        if "error" in resp or not resp.get("success"):
            return []

        posts = []
        for item in resp.get("posts", []):
            posts.append(
                MoltbookPost(
                    id=item.get("id", ""),
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    author=item.get("author", {}).get("name", "unknown"),
                    submolt=item.get("submolt", {}).get("name", "general"),
                    upvotes=item.get("upvotes", 0),
                    comment_count=item.get("comment_count", 0),
                    created_at=item.get("created_at", ""),
                )
            )
        return posts

    def get_post(self, post_id: str) -> dict[str, Any] | None:
        """Fetch a single post."""
        resp = self._request("GET", f"/posts/{post_id}")
        if "error" in resp:
            return None
        return resp.get("post")

    def get_comments(self, post_id: str, limit: int = 20) -> list[MoltbookComment]:
        """Fetch comments on a post."""
        resp = self._request("GET", f"/posts/{post_id}/comments?sort=best&limit={limit}")
        if "error" in resp or not resp.get("success"):
            return []

        comments = []
        for item in resp.get("comments", []):
            comments.append(
                MoltbookComment(
                    id=item.get("id", ""),
                    content=item.get("content", ""),
                    author=item.get("author", {}).get("name", "unknown"),
                    post_id=post_id,
                    upvotes=item.get("upvotes", 0),
                    created_at=item.get("created_at", ""),
                )
            )
        return comments

    def create_post(self, title: str, content: str, submolt: str = "general") -> dict[str, Any] | None:
        """Create a new post."""
        resp = self._request(
            "POST",
            "/posts",
            {
                "title": title,
                "content": content,
                "submolt_name": submolt,
            },
        )
        if "error" in resp:
            return None
        return resp.get("post")

    def create_comment(self, post_id: str, content: str) -> dict[str, Any] | None:
        """Comment on a post."""
        resp = self._request(
            "POST",
            f"/posts/{post_id}/comments",
            {"content": content},
        )
        if "error" in resp:
            return None
        return resp.get("comment")

    def verify_content(self, verification_code: str, answer: str) -> bool:
        """Solve a verification challenge."""
        resp = self._request(
            "POST",
            "/verify",
            {
                "verification_code": verification_code,
                "answer": answer,
            },
        )
        return resp.get("success", False)

    def upvote_post(self, post_id: str) -> bool:
        """Upvote a post."""
        resp = self._request("POST", f"/posts/{post_id}/upvote", {})
        return resp.get("success", False)

    def get_home(self) -> dict[str, Any]:
        """Get home dashboard."""
        resp = self._request("GET", "/home")
        return resp if resp.get("success") else {}

    def get_profile(self) -> dict[str, Any]:
        """Get your profile."""
        resp = self._request("GET", "/agents/me")
        return resp.get("agent", {}) if resp.get("success") else {}
