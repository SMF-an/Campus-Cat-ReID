from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ai.detector import crop_cat_from_bytes
from ai.feature_extractor import CLIPFeatureExtractor
from ai.faiss_index import FaissIndexWrapper
import uvicorn
import os
import torch


app = FastAPI(title="Cat Recognizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize components
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'ai', 'index_data.npz')
index = FaissIndexWrapper(dim=512, path=INDEX_PATH)
index.load()


extractor = CLIPFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu")


@app.post("/identify")
async def identify(file: UploadFile = File(...), location_name: str = Form(""), latitude: float = Form(0.0), longitude: float = Form(0.0)):
    data = await file.read()
    img = crop_cat_from_bytes(data)
    feat = extractor.extract(img)

    scores, ids = index.search(feat.reshape(1, -1), top_k=3)
    top_score = scores[0] if scores else 0.0
    if top_score > 0.90:
        status = "confirmed"
        result = {"status": status, "cat_id": ids[0], "confidence": float(top_score), "candidates": []}
    elif top_score > 0.80:
        status = "uncertain"
        candidates = [{"cat_id": cid, "confidence": float(s)} for cid, s in zip(ids, scores)]
        result = {"status": status, "cat_id": None, "confidence": float(top_score), "candidates": candidates}
    else:
        status = "unknown"
        result = {"status": status, "cat_id": None, "confidence": float(top_score), "candidates": []}

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run("services.identify_service:app", host="127.0.0.1", port=8000, log_level="info")
