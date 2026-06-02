import logging
import os

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai.detector import crop_cat_from_bytes
from ai.faiss_index import FaissIndexWrapper
from ai.feature_extractor import VisionFeatureExtractor


APP_TITLE = "Cat Recognizer"
TOP_K = 3
CONFIRMED_THRESHOLD = 0.66
GAP_THRESHOLD = 0.07
UNCERTAIN_THRESHOLD = 0.55

BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "..", "ai", "index_data.npz")
DEFAULT_CKPT = os.path.join(BASE_DIR, "..", "models", "finetuned_best.pt")

logger = logging.getLogger("cat-identify")
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_services() -> tuple[VisionFeatureExtractor, FaissIndexWrapper]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = VisionFeatureExtractor(device=device)

    checkpoint_path = os.environ.get("MODEL_CHECKPOINT", DEFAULT_CKPT)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            extractor.load_checkpoint(checkpoint_path)
            logger.info("Loaded feature extractor checkpoint: %s (dim=%s)", checkpoint_path, extractor.dim)
        except Exception:
            logger.exception(
                "Failed to load checkpoint '%s'; continuing with pretrained backbone.",
                checkpoint_path,
            )
    else:
        logger.info("No checkpoint found at %s; using pretrained backbone", checkpoint_path)

    index = FaissIndexWrapper(dim=extractor.dim, path=INDEX_PATH)
    index.load()
    if index.vectors.shape[1] != extractor.dim:
        logger.warning(
            "Index dimension %s does not match extractor dimension %s; index was reset and must be rebuilt.",
            index.vectors.shape[1],
            extractor.dim,
        )

    return extractor, index


def _build_candidates(scores: list[float], ids: list[str]) -> list[dict]:
    """Build Top-K candidate list with cat_id and confidence."""
    return [{"cat_id": cid, "confidence": float(score)} for cid, score in zip(ids, scores)]


def build_result(scores: list[float], ids: list[str]) -> dict:
    candidates = _build_candidates(scores, ids)
    if not candidates:
        return {
            "status": "unknown",
            "cat_id": None,
            "confidence": 0.0,
            "candidates": [],
        }

    top_score = candidates[0]["confidence"]
    top2_score = candidates[1]["confidence"] if len(candidates) > 1 else float("-inf")
    score_gap = top_score - top2_score if len(candidates) > 1 else 1.0

    if top_score >= CONFIRMED_THRESHOLD and score_gap >= GAP_THRESHOLD:
        return {
            "status": "confirmed",
            "cat_id": candidates[0]["cat_id"],
            "confidence": top_score,
            "candidates": candidates,  # include Top3 for UI display
        }

    if top_score >= UNCERTAIN_THRESHOLD:
        return {
            "status": "uncertain",
            "cat_id": None,
            "confidence": top_score,
            "candidates": candidates,
        }

    return {
        "status": "unknown",
        "cat_id": None,
        "confidence": top_score,
        "candidates": candidates,  # include Top3 even for unknown, may still be useful
    }


extractor, index = initialize_services()


@app.post("/identify")
async def identify(file: UploadFile = File(...), location_name: str = Form(""), latitude: float = Form(0.0), longitude: float = Form(0.0)):
    image = crop_cat_from_bytes(await file.read())
    feature = extractor.extract(image)
    scores, ids = index.search(feature.reshape(1, -1), top_k=TOP_K)
    return JSONResponse(build_result(scores, ids))


if __name__ == "__main__":
    uvicorn.run("services.identify_service:app", host="127.0.0.1", port=8000, log_level="info")
