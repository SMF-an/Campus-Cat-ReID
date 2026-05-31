from fastapi.testclient import TestClient
from services.identify_service import app
from io import BytesIO
from PIL import Image


def test_identify_endpoint_returns_json():
    client = TestClient(app)
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf, "image/jpeg")}
    resp = client.post("/identify", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
