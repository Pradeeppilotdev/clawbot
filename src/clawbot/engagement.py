from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .agent import ClawbotAgent
from .moltbook_client import MoltbookClient, MoltbookPost


class MoltbookEngagement:
    def __init__(self, agent: ClawbotAgent, moltbook: MoltbookClient, llm=None) -> None:
        self.agent = agent
        self.moltbook = moltbook
        self.llm = llm  # Optional direct LLM for quick replies
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

        # Use direct LLM if available for fast replies, otherwise use agent
        if self.llm:
            answer = self._get_direct_llm_response(
                f"""You're a builder chatting on Moltbook (like Reddit for AI builders). Reply naturally to this post, like you're talking to a peer.

Post: "{post.title}"
{post.content}

Rules:
- Sound human and casual
- NO markdown (no **, ##, etc)
- NO "Summary" or "Action Plan"
- 2-3 sentences max
- Share experience or ask a follow-up"""
            )
        else:
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

        prompt = f"""Write a post for Moltbook about: {topic}

Format your response EXACTLY like this:
TITLE: [catchy title under 80 chars]
CONTENT: [your post content]

Rules:
- The title should be a hook or hot take, not generic
- Content should be 100-200 words, conversational
- Sound like a dev sharing real experience, not a corporate blog
- NO markdown (no ** or ## or bullets)
- NO preamble like "Here's" or "Sure" - just the TITLE: and CONTENT:
- Be opinionated, share what actually works"""

        result = self.agent.run(prompt)
        raw = result.answer.strip()

        if not raw or len(raw) < 80:
            print(f"    ⚠️ Content too short")
            return False

        # Parse TITLE: and CONTENT: format
        title, content = self._parse_post_format(raw)

        if not title or not content:
            print(f"    ⚠️ Could not parse title/content format")
            return False

        # Post it - use agents submolt for agent-related content
        post_resp = self.moltbook.create_post(
            title=title,
            content=content,
            submolt="agents",  # Post to m/agents instead of m/general
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
            "posts_upvoted": 0,
            "insights_posted": 0,
            "errors": 0,
        }

        try:
            # Get feed
            posts = self.moltbook.get_feed(limit=limit, sort="hot")
            stats["posts_checked"] = len(posts)
            print(f"  📬 Fetched {len(posts)} posts from feed")

            # Upvote good posts about agents/AI
            for post in posts[:5]:  # Only check top 5
                if self._should_upvote(post):
                    if self.moltbook.upvote_post(post.id):
                        stats["posts_upvoted"] += 1
                        print(f"  👍 Upvoted: {post.title[:50]}...")

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
                # Topics specific to Clawbot's expertise: hackathon agents, failover, tools
                topics = [
                    "why your hackathon agent needs a fallback LLM provider",
                    "the planner-research-writer pipeline for fast agent prototypes",
                    "building a tool registry pattern that actually scales",
                    "local data tools vs API calls - when to use which",
                    "scoring agent outputs objectively with scorecards",
                    "circuit breakers saved my demo - here's how",
                    "stop overengineering your hackathon agent",
                    "the 3 tools every hackathon agent actually needs",
                ]
                topic = topics[hash(str(__import__("time").time())) % len(topics)]
                if self.generate_insight(topic):
                    stats["insights_posted"] += 1

        except Exception as exc:
            print(f"  ❌ Error during engagement: {exc}")
            stats["errors"] += 1

        self._save_state()

        print(f"\n📊 Summary: {stats['posts_upvoted']} upvotes, {stats['posts_responded']} replies, {stats['insights_posted']} insights")
        return stats

    def _should_upvote(self, post: MoltbookPost) -> bool:
        """Decide if post deserves an upvote."""
        # Skip own posts
        if self.own_agent_name and post.author == self.own_agent_name:
            return False

        # Skip already engaged posts (we probably already upvoted)
        if post.id in self.engaged_post_ids:
            return False

        text = f"{post.title} {post.content}".lower()

        # Upvote posts about agent building, hackathons, tools
        agent_keywords = ["agent", "llm", "hackathon", "tool", "pipeline", "failover", "prompt"]
        return any(k in text for k in agent_keywords)

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
    def _parse_post_format(raw: str) -> tuple[str, str]:
        """Parse TITLE: and CONTENT: format from LLM output."""
        title = ""
        content = ""

        # Try to extract TITLE: and CONTENT:
        lines = raw.strip().split("\n")
        in_content = False
        content_lines = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.upper().startswith("TITLE:"):
                title = line_stripped[6:].strip().strip('"').strip("'")
            elif line_stripped.upper().startswith("CONTENT:"):
                in_content = True
                rest = line_stripped[8:].strip()
                if rest:
                    content_lines.append(rest)
            elif in_content:
                content_lines.append(line)

        content = "\n".join(content_lines).strip()

        # Fallback: if no TITLE:/CONTENT: format, use first line as title
        if not title and not content:
            lines = raw.strip().split("\n")
            # Strip common prefixes
            first_line = lines[0] if lines else ""
            for prefix in ["here's", "sure,", "here is", "okay,", "alright,"]:
                if first_line.lower().startswith(prefix):
                    first_line = first_line[len(prefix):].strip()
                    # Also strip "the moltbook post:" etc
                    for suffix in ["the moltbook post:", "a moltbook post:", "the post:", "a post:"]:
                        if first_line.lower().startswith(suffix):
                            first_line = first_line[len(suffix):].strip()
                    break
            title = first_line[:100] if first_line else "Quick thought on agent building"
            content = raw.strip()

        return title, content

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

    def _get_direct_llm_response(self, prompt: str) -> str:
        """Get a direct response from LLM without full agent pipeline."""
        if not self.llm:
            return ""
        try:
            response = self.llm.complete("You are a natural conversationalist on Moltbook.", prompt)
            return response.strip() if response else ""
        except Exception:
            return ""
