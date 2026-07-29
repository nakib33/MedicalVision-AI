"""MedicalVision AI Suite - Main FastAPI Application.

Serves all 12 medical imaging projects from a single web interface.
Each project has its own inference, XAI explainability, and reporting endpoints.
"""

import io
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from PIL import Image

from projects import discover_projects, get_projects_summary
from shared.config import DEVICE
from shared.pipelines.inference import predict
from shared.pipelines.transforms import get_inference_transform
from shared.explainers.xai_factory import run_all_explainers
from shared.utils.report import generate_prediction_report

# ═══ App Initialisation ═══════════════════════════════════════
app = FastAPI(
    title="MedicalVision AI Suite",
    description="12 CNN-based medical imaging projects with Explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

templates = Jinja2Templates(directory="shared/templates")
app.mount("/static", StaticFiles(directory="shared/static"), name="static")

# Storage for last prediction per project (for report generation)
_last_images: dict = {}
_last_predictions: dict = {}
_last_explanations: dict = {}

# Discover all 12 projects
projects = discover_projects()
print(f"\n{'='*60}")
print(f"  MedicalVision AI Suite - {len(projects)} Projects Loaded")
print(f"{'='*60}")
for p in projects:
    print(f"  {p['id']}: {p['name']} ({len(p['classes'])} classes, {p['model']})")
print(f"{'='*60}\n")


# ═══ Helpers ══════════════════════════════════════════════════
async def load_uploaded_image(file: UploadFile) -> Image.Image:
    """Validate and load an uploaded image file into a PIL Image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported (JPEG, PNG, BMP, TIFF).")
    contents = await file.read()
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image. Upload a valid medical image file.")


# ═══ Routes: Home ═════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page: project grid overview."""
    return templates.TemplateResponse(
        request,
        "home.html",
        {"projects": get_projects_summary()},
    )


@app.get("/api/projects")
async def api_projects():
    """Return all projects as JSON."""
    return {"projects": get_projects_summary()}


# ═══ Routes: About ════════════════════════════════════════════
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page."""
    return templates.TemplateResponse(
        request,
        "about.html",
        {"projects": get_projects_summary()},
    )


# ═══ Health Check ═════════════════════════════════════════════
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "projects_loaded": len(projects),
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
    }


# ═══ Project Route Registration ══════════════════════════════
def register_project_routes(project: dict):
    """Register GET/POST routes for a single project.

    Creates route handlers that capture this project's config and loader
    in a closure, so each project behaves independently.
    """
    pid = project["id"]
    cfg = project["config_module"]
    loader = project["loader_module"]

    # Mount project-specific static files
    static_dir = os.path.join("projects", pid, "static")
    if os.path.isdir(static_dir):
        app.mount(f"/{pid}/static", StaticFiles(directory=static_dir), name=f"s_{pid}")

    # ── GET /{pid}/  – Project UI page ─────────────────────
    @app.get(f"/{pid}/", response_class=HTMLResponse)
    async def project_ui(request: Request):
        proj_info = {
            "id": pid,
            "name": project["name"],
            "short_name": project["short_name"],
            "description": project["description"],
            "classes": project["classes"],
            "model": project["model"],
            "img_size": project["img_size"],
        }
        return templates.TemplateResponse(
            request,
            "project_ui.html",
            {"project": proj_info},
        )

    # ── POST /{pid}/predict – Prediction ───────────────────
    @app.post(f"/{pid}/predict")
    async def predict_endpoint(file: UploadFile = File(...)):
        try:
            image = await load_uploaded_image(file)
            model = loader.get_model()

            result = predict(
                model=model,
                image=image,
                class_names=project["classes"],
                device=DEVICE,
                img_size=project["img_size"],
            )

            # Store for report generation
            _last_images[pid] = image
            _last_predictions[pid] = result

            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Prediction failed: {str(e)}")

    # ── POST /{pid}/explain – XAI Explanations ───────────
    @app.post(f"/{pid}/explain")
    async def explain_endpoint(file: UploadFile = File(...)):
        try:
            image = await load_uploaded_image(file)
            model = loader.get_model()

            transform = get_inference_transform(project["img_size"])
            tensor = transform(image).unsqueeze(0).to(DEVICE)

            result = run_all_explainers(
                model=model,
                image_tensor=tensor,
                class_names=project["classes"],
                device=DEVICE,
                occlude_size=min(32, project["img_size"] // 7),
                occlude_stride=min(16, project["img_size"] // 14),
            )

            _last_images[pid] = image
            _last_predictions[pid] = result["predictions"]
            _last_explanations[pid] = result["explanations"]

            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"XAI analysis failed: {str(e)}")

    # ── GET /{pid}/report – Download PDF Report ───────────
    @app.get(f"/{pid}/report")
    async def download_report():
        image = _last_images.get(pid)
        pred = _last_predictions.get(pid)
        explanations = _last_explanations.get(pid) or {}

        if image is None or pred is None:
            raise HTTPException(
                400, "No prediction available. Upload an image and run analysis first."
            )

        try:
            pdf_bytes = generate_prediction_report(
                project_name=project["name"],
                class_names=project["classes"],
                predicted_class=pred.get("predicted_class", "Unknown"),
                confidence=pred.get("confidence", 0.0),
                probabilities=pred.get("probabilities", {}),
                uploaded_image=image,
                explanations=explanations,
            )

            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{pid}_report.pdf"'
                },
            )
        except Exception as e:
            raise HTTPException(500, f"Report generation failed: {str(e)}")


# Register routes for all projects
for proj in projects:
    register_project_routes(proj)
