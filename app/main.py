"""
FastAPI application factory.

Key production concerns addressed here:
- Structured startup/shutdown logging
- CORS configured (open by default — tighten in production)
- Global exception handler returns consistent JSON error responses
- /docs and /redoc available only when DEBUG=True
- Request timing middleware for performance monitoring
"""

from __future__ import annotations

import time
import traceback

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.api.routes import router
from app.models.response import ErrorResponse
from app.utils.logger import setup_logging, get_logger

# ---------------------------------------------------------------------------
# Logging must be configured before the first `get_logger` call
# ---------------------------------------------------------------------------
setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        # Disable interactive docs in production
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
    )

    # ---- CORS ---------------------------------------------------------------
    # Tighten allow_origins in production (e.g. ["https://yourfrontend.com"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request timing middleware -------------------------------------------
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
        logger.debug(
            "Request processed",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response

    # ---- Global exception handlers ------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Request validation error", extra={"detail": exc.errors()})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                success=False,
                message="Request validation failed",
                detail=str(exc.errors()),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            extra={"traceback": traceback.format_exc()},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                success=False,
                message="An unexpected server error occurred",
                detail=str(exc) if settings.DEBUG else None,
            ).model_dump(),
        )

    # ---- Routes -------------------------------------------------------------
    app.include_router(router, prefix="/api/v1")

    # ---- Startup / Shutdown events ------------------------------------------

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info(
            f"{settings.APP_NAME} v{settings.APP_VERSION} starting up",
            extra={"debug": settings.DEBUG, "ocr_lang": settings.OCR_LANG},
        )

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info(f"{settings.APP_NAME} shutting down")

    # ---- Root redirect to docs (debug) or simple info (prod) ----------------

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
            "scan_endpoint": "/api/v1/scan-slip",
        }

    return app


# ---------------------------------------------------------------------------
# Application instance (used by uvicorn)
# ---------------------------------------------------------------------------

app = create_app()
