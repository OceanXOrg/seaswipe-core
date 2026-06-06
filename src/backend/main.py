"""FastAPI application — serves the analysis API and the built frontend.

The app is created by ``create_app()`` and run in-process by
``seaswipe_analysis.py`` (which sets the active Workspace first). Because the
Workspace is a module-level singleton, the server must be run without reload /
multiple workers.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.backend.api import analysis

logger = logging.getLogger(__name__)

# Repo root: .../seaswipe-open-source ; built frontend lives at frontend/dist.
REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="SeaSwipe (open source)", version="1.0.0")

    # CORS so the Vite dev server (localhost:3000) can talk to the API during
    # frontend development. In production the frontend is served from this same
    # origin, so this is only relevant for `npm run dev`.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(analysis.router, prefix="/api", tags=["analysis"])

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # Serve the built single-page frontend if it exists. When it doesn't (the
    # user hasn't run `npm run build`), return a helpful message at "/".
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    else:
        @app.get("/")
        def no_frontend():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Frontend not built.",
                    "hint": "Run `cd frontend && npm install && npm run build`, "
                            "or use `npm run dev` for development.",
                },
            )

    return app


app = create_app()
