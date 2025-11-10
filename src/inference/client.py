"""
Client for interacting with vLLM server for inference.

Provides high-level API for generating JSON outputs with constrained decoding.
"""

import json
from typing import Optional, Dict, Any

import httpx
from pydantic import ValidationError

from .schemas import BranchInfo, GenerateRequest


class InferenceClient:
    """Client for Qwen3-8B inference via vLLM server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def generate(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 200,
        use_constrained_decoding: bool = True,
    ) -> BranchInfo:
        """
        Generate title and branch name from prompt.

        Args:
            prompt: User's task description
            model: Model adapter name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_constrained_decoding: Use JSON schema constraints

        Returns:
            BranchInfo with title and branch_name

        Raises:
            httpx.HTTPError: If request fails
            ValidationError: If response doesn't match schema
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates JSON with a title and git branch name."
            },
            {"role": "user", "content": prompt}
        ]

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add constrained decoding if enabled
        if use_constrained_decoding:
            request_data["extra_body"] = {
                "guided_json": BranchInfo.model_json_schema()
            }

        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_data
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse and validate JSON
        try:
            data = json.loads(content)
            return BranchInfo.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Invalid response format: {content}") from e

    def generate_raw(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 200,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate raw text without schema validation.

        Useful for debugging or custom use cases.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_data
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def list_models(self) -> Dict[str, Any]:
        """List available models on the server."""
        response = self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncInferenceClient:
    """Async client for Qwen3-8B inference via vLLM server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 200,
        use_constrained_decoding: bool = True,
    ) -> BranchInfo:
        """Async version of generate."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates JSON with a title and git branch name."
            },
            {"role": "user", "content": prompt}
        ]

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if use_constrained_decoding:
            request_data["extra_body"] = {
                "guided_json": BranchInfo.model_json_schema()
            }

        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_data
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        try:
            data = json.loads(content)
            return BranchInfo.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Invalid response format: {content}") from e

    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
