"""Centralized configuration for the summarizer project."""

SYSTEM_PROMPT = """You are a careful assistant that outputs ONLY valid JSON matching the schema:
{
  "summary": "<2-4 words, Title Case, no punctuation>",
  "branch": "<kebab-case, lowercase, [a-z0-9-] only, max 3 words, prefix with a category like bug/, feat/, etc.>"
}
Never include explanations or extra keys.

Turn this request for code changes into:
1) a 2-4 word summary (Title Case),
2) a friendly git branch name (prefixed kebab-case).
"""

SYSTEM_PROMPT_TWO_LINE = """You are a careful assistant that generates task summaries and git branch names.

Output EXACTLY two lines with no other text:
Line 1: A 2-4 word summary (Title Case, no punctuation)
Line 2: A git branch name (kebab-case, lowercase, [a-z0-9-] only, max 4 words, prefix with a category like bug/, feat/, etc.)

Examples:
Fix Login Bug
bug/fix-login

Add Dark Mode
feat/add-dark-mode

API Docs
docs/api-polish

Refactor User Service V2
chore/user-service-v2-refactor

Turn this request for code changes into:
1) a 2-4 word summary (Title Case),
2) a friendly git branch name (prefixed kebab-case).
"""
