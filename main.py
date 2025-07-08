from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pyannote.audio import Inference
from torch.nn.functional import cosine_similarity
import torch
import os
import shutil
from pyannote.audio import Model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear carpeta si no existe
os.makedirs("embeddings", exist_ok=True)



HF_TOKEN = os.environ["HF_TOKEN"]

model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
inference = Inference(model, window="whole")


@app.post("/register")
async def register_user(nombre: str = Form(...), file: UploadFile = File(...)):
    path_temp = "temp.wav"
    with open(path_temp, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = inference(path_temp)
    os.remove(path_temp)

    torch.save(embedding, f"embeddings/{nombre}.vec")
    return {"mensaje": f"Usuario '{nombre}' registrado"}

@app.post("/identify")
async def identify_user(file: UploadFile = File(...)):
    path_temp = "temp.wav"
    with open(path_temp, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = inference(path_temp)
    os.remove(path_temp)

    best_score = -1.0
    best_user = "desconocido"

    for fname in os.listdir("embeddings"):
        ref = torch.load(f"embeddings/" + fname)
        score = cosine_similarity(embedding, ref, dim=0).item()
        if score > best_score:
            best_score = score
            best_user = fname.replace(".vec", "")

    return {"usuario": best_user, "score": round(best_score, 4)}
