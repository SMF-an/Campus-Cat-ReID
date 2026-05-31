# Campus Cat Recognition — AI module scaffold

This folder contains a scaffold for the cat recognition pipeline (detection → CLIP features → FAISS retrieval) and a minimal FastAPI identify service.

Quick start (create virtualenv and install):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn services.identify_service:app --reload
```

Notes:
- The code contains fallbacks when heavy dependencies are not available (dummy vectors, numpy brute-force index).
- Populate `ai/index_data.npz` with vectors and ids before expecting meaningful identification results.
