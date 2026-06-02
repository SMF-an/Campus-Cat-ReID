from PIL import Image
from ultralytics import YOLO
from io import BytesIO


def crop_cat_from_bytes(image_bytes: bytes) -> Image.Image:
    """Attempt to detect and crop a cat from image bytes. If detection backend missing,
    returns the original image as a PIL.Image."""
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        yolo = YOLO("yolov8m.pt")
        results = yolo(img)
        boxes = results[0].boxes
        for box in boxes:
            # COCO class 15 == cat
            if int(box.cls) == 15:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                return img.crop((x1, y1, x2, y2))
        return img
    except Exception:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
