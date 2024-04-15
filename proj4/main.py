# STEP 1
from transformers import pipeline

# STEP 2
vision_classifier = pipeline(model="google/vit-base-patch16-224")

from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from PIL import Image
import io
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import uvicorn


# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app = FaceAnalysis(name='buffalo_l',providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


app = FastAPI()

@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(files: list[UploadFile]):
    faces = []
    for file in files:
        contents = await file.read()
        ##### OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces.append(face_app.get(img))

    feat1 = np.array(faces[0][0].normed_embedding, dtype=np.float32)
    feat2 = np.array(faces[1][0].normed_embedding, dtype=np.float32)
    sims = np.dot(feat1, feat2)

    return {"result": float(sims)}
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")