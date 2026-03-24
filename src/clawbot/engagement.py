from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .agent import ClawbotAgent
from .moltbook_client import MoltbookClient, MoltbookPost


class MoltbookEngagement:
    def __init__(self, agent: ClawbotAgent, moltbook: MoltbookClient) -> None:
        self.agent = agent
        self.moltbook = moltbook
        self.state_path = Path("runs/moltbook_state.json")
        self.engaged_post_ids: set[str] = set()
        self.recent_reply_hashes: list[str] = []
        self.own_agent_name = ""
        self._load_state()
        self._load_profile()

    def _load_profile(self) -> None:
        profile = self.moltbook.get_profile()
        name = str(profile.get("name", "")).strip()
        if name:
            self.own_agent_name = name

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                ids = data.get("engaged_post_ids", [])
                if isinstance(ids, list):
                    self.engaged_post_ids = {str(x) for x in ids}
                hashes = data.get("recent_reply_hashes", [])
                if isinstance(hashes, list):
                    self.recent_reply_hashes = [str(x) for x in hashes[:30]]
        except Exception:
            pass

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engaged_post_ids": sorted(self.engaged_post_ids),
            "recent_reply_hashes": self.recent_reply_hashes[:30],
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def should_respond_to_post(self, post: MoltbookPost) -> bool:
        """Heuristic: decide if post warrants a response."""
        text = f"{post.title} {post.content}".lower()

        # Skip if already engaged
        if post.id in self.engaged_post_ids:
            return False

        # Skip own posts
        if self.own_agent_name and post.author == self.own_agent_name:
            return False

        # Skip if we've already commented on this post in previous runs.
        if self._already_commented(post.id):
            self.engaged_post_ids.add(post.id)
            return False

        # Respond to agent/AI/hackathon questions
        keywords = ["agent", "ai ", "llm", "model", "autonomous", "hackathon", "design", "build", "how do", "advice"]
        has_keywords = any(k in text for k in keywords)
        has_question = "?" in post.content

        return has_keywords and has_question

    def _already_commented(self, post_id: str) -> bool:
        if post_id in self.engaged_post_ids:
            return True
        comments = self.moltbook.get_comments(post_id, limit=50)
        if not comments:
            return False
        for comment in comments:
            if self.own_agent_name and comment.author == self.own_agent_name:
                return True
        return False

    def respond_to_post(self, post: MoltbookPost) -> bool:
        """Run pipeline and respond with thoughtful comment."""
        print(f"  📝 Responding to: {post.title}")

        # Run agent pipeline on post - sound like a real person, not a bot
        result = self.agent.run(f"""You're a developer chatting on Moltbook (like Reddit for AI builders). Reply to this post naturally, like you're talking to a friend.

Post: "{post.title}"
{post.content}

Rules:
- Sound human and casual, not like a corporate AI
- NO markdown formatting (no ** or ## or bullet points)
- NO "Summary" or "Action Plan" headers
- Just 2-3 sentences max, like a real comment
- Share a quick opinion, experience, or ask a follow-up question
- Match the vibe of the post (if they're being casual/funny, you can be too)""")

        answer = result.answer.strip()
        if not answer or len(answer) < 40:
            print(f"    ⚠️ Answer too short, skipping")
            return False

        if self._looks_generic_or_outage(answer):
            print(f"    ⚠️ Skipping generic fallback response")
            return False

        normalized_hash = self._reply_hash(answer)
        if normalized_hash in self.recent_reply_hashes:
            print(f"    ⚠️ Skipping duplicate-style response")
            return False

        # Post comment
        comment_resp = self.moltbook.create_comment(post.id, answer)
        if not comment_resp:
            print(f"    ❌ Failed to post comment")
            return False

        # Check for verification challenge
        if comment_resp.get("verification"):
            verification = comment_resp["verification"]
            challenge = verification.get("challenge_text", "")
            code = verification.get("verification_code", "")

            # Try to solve challenge
            challenge_answer = self._solve_challenge(challenge)
            if challenge_answer:
                self.moltbook.verify_content(code, challenge_answer)
                print(f"    ✅ Comment posted and verified!")
            else:
                print(f"    ⚠️ Verification challenge, may be pending")
        else:
            print(f"    ✅ Comment posted!")

        self.recent_reply_hashes.insert(0, normalized_hash)
        self.recent_reply_hashes = self.recent_reply_hashes[:30]
        self.engaged_post_ids.add(post.id)
        return True

    def generate_insight(self, topic: str = "agent design") -> bool:
        """Proactively generate and post an insight."""
        print(f"  💡 Generating insight about: {topic}")

        prompt = f"""Write a casual post for Moltbook (Reddit-style AI builder community) about {topic}.

Rules:
- Sound like a real developer sharing thoughts, not a corporate blog
- NO markdown (no ** or ## or bullets)
- Start with a hook or hot take, not "I've been thinking about..."
- Share a real opinion or lesson learned
- Keep it under 200 words, conversational
- Can be a bit spicy or have personality"""

        result = self.agent.run(prompt)
        content = result.answer.strip()

        if not content or len(content) < 80:
            print(f"    ⚠️ Content too short")
            return False

        # Extract title from first line or generate one
        lines = content.split("\n")
        title = lines[0][:100] if lines else "Thoughts on AI Agents"

        # Post it
        post_resp = self.moltbook.create_post(
            title=title,
            content=content,
            submolt="general",
        )

        if not post_resp:
            print(f"    ❌ Failed to create post")
            return False

        # Check for verification
        if post_resp.get("verification"):
            verification = post_resp["verification"]
            challenge = verification.get("challenge_text", "")
            code = verification.get("verification_code", "")

            challenge_answer = self._solve_challenge(challenge)
            if challenge_answer:
                self.moltbook.verify_content(code, challenge_answer)
                print(f"    ✅ Post published and verified!")
            else:
                print(f"    ⚠️ Verification challenge, may be pending")
        else:
            print(f"    ✅ Post published!")

        return True

    def engage_batch(self, limit: int = 10, respond: bool = True, post_insight: bool = False) -> dict[str, Any]:
        """Run one batch of engagement."""
        print(f"\n🦞 Clawbot Engagement Check")
        print(f"  Time: {__import__('datetime').datetime.now().isoformat()}")

        stats = {
            "posts_checked": 0,
            "posts_responded": 0,
            "insights_posted": 0,
            "errors": 0,
        }

        try:
            # Get feed
            posts = self.moltbook.get_feed(limit=limit, sort="hot")
            stats["posts_checked"] = len(posts)
            print(f"  📬 Fetched {len(posts)} posts from feed")

            # Respond to relevant posts
            if respond:
                max_replies_per_run = 2
                for post in posts:
                    if stats["posts_responded"] >= max_replies_per_run:
                        break
                    if self.should_respond_to_post(post):
                        if self.respond_to_post(post):
                            stats["posts_responded"] += 1

            # Optionally post an insight
            if post_insight:
                topics = [
                    "agent reliability and error handling",
                    "building effective tool implementations",
                    "balancing speed vs accuracy in agent design",
                    "strategies for multi-LLM failover",
                ]
                topic = topics[hash(str(__import__("time").time())) % len(topics)]
                if self.generate_insight(topic):
                    stats["insights_posted"] += 1

        except Exception as exc:
            print(f"  ❌ Error during engagement: {exc}")
            stats["errors"] += 1

        self._save_state()

        print(f"\n📊 Summary: {stats['posts_responded']} replies, {stats['insights_posted']} insights posted")
        return stats

    @staticmethod
    def _reply_hash(text: str) -> str:
        compact = re.sub(r"\s+", " ", text.strip().lower())
        compact = re.sub(r"[^a-z0-9 ]", "", compact)
        return compact[:220]

    @staticmethod
    def _looks_generic_or_outage(text: str) -> bool:
        lower = text.lower()
        markers = [
            "all configured llm providers are temporarily unavailable",
            "llm_provider_error",
            "deterministic fallback logic",
            "mock mode is active",
            "external model providers are currently rate-limited",
        ]
        return any(m in lower for m in markers)

    @staticmethod
    def _solve_challenge(challenge_text: str) -> str | None:
        """Attempt to solve Moltbook math verification challenge."""
        try:
            # Clean up challenge text
            clean = re.sub(r"[^a-zA-Z0-9\s\+\-\*/\.]", " ", challenge_text).lower()

            # Extract numbers and operation
            numbers = re.findall(r"\d+(?:\.\d+)?", clean)
            if len(numbers) < 2:
                return None

            num1, num2 = float(numbers[0]), float(numbers[1])

            # Detect operation (look for keywords in original text)
            text_lower = challenge_text.lower()
            if "plus" in text_lower or "add" in text_lower or "+" in text_lower:
                result = num1 + num2
            elif "minus" in text_lower or "subtract" in text_lower or "-" in text_lower:
                result = num1 - num2
            elif "times" in text_lower or "multiply" in text_lower or "*" in text_lower:
                result = num1 * num2
            elif "divide" in text_lower or "/" in text_lower:
                result = num1 / num2 if num2 != 0 else 0
            else:
                # Default to addition
                result = num1 + num2

            return f"{result:.2f}"
        except Exception:
            return None
