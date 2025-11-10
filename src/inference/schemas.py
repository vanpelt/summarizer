"""
Pydantic schemas for JSON output validation and constrained decoding.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
import re


class BranchInfo(BaseModel):
    """Schema for git branch information generation."""

    title: str = Field(
        ...,
        description="Short, concise description of the task (3-7 words)",
        min_length=5,
        max_length=100,
        examples=["Add User Authentication", "Fix Memory Leak in Pipeline"]
    )

    branch_name: str = Field(
        ...,
        description="Git branch name in kebab-case with type prefix",
        pattern=r'^(feat|fix|docs|test|refactor|perf|chore)\/[a-z0-9-]+$',
        examples=["feat/user-auth", "fix/memory-leak", "docs/api-update"]
    )

    @field_validator('branch_name')
    @classmethod
    def validate_branch_name(cls, v: str) -> str:
        """Validate branch name format."""
        # Check prefix
        valid_prefixes = ['feat', 'fix', 'docs', 'test', 'refactor', 'perf', 'chore']
        prefix = v.split('/')[0] if '/' in v else ''

        if prefix not in valid_prefixes:
            raise ValueError(
                f"Branch name must start with one of: {', '.join(valid_prefixes)}"
            )

        # Check format (lowercase, kebab-case)
        if not re.match(r'^[a-z]+\/[a-z0-9-]+$', v):
            raise ValueError(
                "Branch name must be lowercase kebab-case with a type prefix (e.g., feat/my-feature)"
            )

        return v

    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate title is concise and well-formatted."""
        # Strip extra whitespace
        v = ' '.join(v.split())

        # Check word count (approximately)
        word_count = len(v.split())
        if word_count < 2:
            raise ValueError("Title must contain at least 2 words")
        if word_count > 10:
            raise ValueError("Title should be concise (max 10 words)")

        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Add User Authentication",
                    "branch_name": "feat/user-auth"
                },
                {
                    "title": "Fix Memory Leak in Image Pipeline",
                    "branch_name": "fix/image-memory-leak"
                },
                {
                    "title": "Update API Documentation",
                    "branch_name": "docs/api-update"
                }
            ]
        }
    }


class GenerateRequest(BaseModel):
    """Request schema for generation endpoint."""

    prompt: str = Field(
        ...,
        description="User's task description",
        min_length=3,
        max_length=500,
        examples=["Add dark mode to settings", "Fix the login bug"]
    )

    model: str = Field(
        default="default",
        description="Model adapter to use for generation",
        examples=["model1", "model2", "default"]
    )

    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic)"
    )

    max_tokens: Optional[int] = Field(
        default=200,
        ge=10,
        le=1000,
        description="Maximum number of tokens to generate"
    )


class GenerateResponse(BaseModel):
    """Response schema for generation endpoint."""

    title: str
    branch_name: str
    model: str = Field(description="Model that generated this response")
    prompt: str = Field(description="Original prompt")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    models_loaded: int
    vllm_available: bool


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    name: str
    adapter_path: Optional[str] = None
    base_model: str = "Qwen/Qwen3-8B"
    loaded: bool = True


class ModelsListResponse(BaseModel):
    """Response for listing available models."""

    models: list[ModelInfo]
    count: int
