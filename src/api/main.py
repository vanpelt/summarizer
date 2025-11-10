"""
FastAPI application for Qwen3-8B JSON generation.

Provides REST API endpoints for generating title and branch names.
"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..inference.client import AsyncInferenceClient
from ..inference.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
)


# Global client instance
client: AsyncInferenceClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the API."""
    global client

    # Startup: initialize inference client
    vllm_url = "http://localhost:8000"
    client = AsyncInferenceClient(base_url=vllm_url)

    yield

    # Shutdown: cleanup
    if client:
        await client.close()


# Create FastAPI app
app = FastAPI(
    title="Qwen3-8B JSON Generator API",
    description="Generate git branch names and titles from task descriptions",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Qwen3-8B JSON Generator API",
        "version": "0.1.0",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "models": "/models",
            "docs": "/docs",
        }
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate title and branch name from a task description.

    Args:
        request: GenerateRequest with prompt and optional model selection

    Returns:
        GenerateResponse with title, branch_name, model, and original prompt

    Raises:
        HTTPException: If generation fails
    """
    try:
        result = await client.generate(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_constrained_decoding=True,
        )

        return GenerateResponse(
            title=result.title,
            branch_name=result.branch_name,
            model=request.model,
            prompt=request.prompt,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns:
        HealthResponse with server status
    """
    vllm_healthy = await client.health_check() if client else False

    # Try to get model count
    models_count = 0
    if vllm_healthy:
        try:
            models_data = await client.list_models()
            models_count = len(models_data.get("data", []))
        except Exception:
            pass

    status = "healthy" if vllm_healthy else "unhealthy"

    return HealthResponse(
        status=status,
        models_loaded=models_count,
        vllm_available=vllm_healthy,
    )


@app.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List available models.

    Returns:
        ModelsListResponse with list of available models
    """
    try:
        models_data = await client.list_models()
        model_list = []

        for model in models_data.get("data", []):
            model_info = ModelInfo(
                name=model.get("id", "unknown"),
                base_model=model.get("owned_by", "Qwen/Qwen3-8B"),
            )
            model_list.append(model_info)

        return ModelsListResponse(
            models=model_list,
            count=len(model_list),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
