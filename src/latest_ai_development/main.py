#!/usr/bin/env python
import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from latest_ai_development.tracking import initialize_langsmith_tracing

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
logger = logging.getLogger("uvicorn.error")


def _normalize_context(raw_context: str | None) -> str:
    """Return a stable path prefix like '/testing' or ''."""
    if not raw_context:
        return ""

    normalized = raw_context.strip()
    if not normalized or normalized == "/":
        return ""

    normalized = "/" + normalized.strip("/")
    return normalized


CONTEXT_PATH = _normalize_context(os.getenv("CONTEXT"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_langsmith_tracing()
    yield


app = FastAPI(root_path=CONTEXT_PATH, lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


class QueryRequest(BaseModel):
    natural_language_query: str
    user_identity: str
    prior_query_context: Optional[str] = None


def run():
    """
    Run the crew.
    """
    inputs = {
        "natural_language_query": "Summarize approved baby monitor inventory across categories.",
        "user_identity": "retail_manager_north",
        "prior_query_context": None,
    }
    initialize_langsmith_tracing()
    from latest_ai_development.crew import LatestAiDevelopment

    try:
        LatestAiDevelopment().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "natural_language_query": "Generate an inventory summary for baby carriers.",
        "user_identity": "analyst_west",
        "prior_query_context": None,
    }
    initialize_langsmith_tracing()
    from latest_ai_development.crew import LatestAiDevelopment

    try:
        LatestAiDevelopment().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    initialize_langsmith_tracing()
    from latest_ai_development.crew import LatestAiDevelopment

    try:
        LatestAiDevelopment().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "natural_language_query": "Provide pricing trends for top baby strollers this quarter.",
        "user_identity": "analyst_east",
        "prior_query_context": None,
    }
    initialize_langsmith_tracing()
    from latest_ai_development.crew import LatestAiDevelopment

    try:
        LatestAiDevelopment().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "natural_language_query": "",
        "user_identity": "",
        "prior_query_context": None,
    }
    initialize_langsmith_tracing()
    from latest_ai_development.crew import LatestAiDevelopment

    try:
        result = LatestAiDevelopment().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")


@app.post("/ask")
async def ask(request: QueryRequest):
    inputs = {
        "natural_language_query": request.natural_language_query,
        "user_identity": request.user_identity,
        "prior_query_context": request.prior_query_context,
    }
    try:
        from latest_ai_development.crew import LatestAiDevelopment

        result = LatestAiDevelopment().crew().kickoff(inputs=inputs)
        return {"status": "success", "response": result}
    except Exception as e:
        logger.error(f"Error occurred while asking the crew: {e}")
        content = {
            "status": "error",
            "response": "Something went wrong. Please contact your administrator.",
        }
        return JSONResponse(status_code=500, content=content)


if CONTEXT_PATH:
    app.add_api_route(
        f"{CONTEXT_PATH}/health",
        health_check,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{CONTEXT_PATH}/ask",
        ask,
        methods=["POST"],
        include_in_schema=False,
    )


if __name__ == "__main__":
    uvicorn.run("latest_ai_development.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
