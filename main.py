from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
import torch
from pyannote.audio import Inference
from torch.nn.functional import cosine_similarity

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
inference = Inference("pyannote/embedding", use_auth_token="TU_TOKEN_DE_HF")

# Crear carpeta embeddings si no existe
os.makedirs("embeddings", exist_ok=True)

@app.post("/register")
async def register_user(name: str = Query(...), file: UploadFile = File(...)):
    """
    Registrar un nuevo usuario de voz
    """
    path_temp = "temp_register.wav"
    with open(path_temp, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    emb = inference(path_temp).data  # .data → extrae el Tensor
    os.remove(path_temp)

    path_embedding = os.path.join("embeddings", f"{name}.pt")
    torch.save(emb, path_embedding)

    return {"status": "ok", "usuario": name}

@app.post("/identify")
async def identify_user(file: UploadFile = File(...)):
    """
    Identificar quién habla
    """
    temp_path = "temp_identify.wav"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = inference(temp_path).data  # SlidingWindowFeature → Tensor
    os.remove(temp_path)

    best_score = -1.0
    best_user = "desconocido"
    for fname in os.listdir("embeddings"):
        ref_path = os.path.join("embeddings", fname)
        if not fname.endswith(".pt"):
            continue
        ref = torch.load(ref_path)
        score = cosine_similarity(embedding, ref, dim=0).item()
        if score > best_score:
            best_score = score
            best_user = fname.replace(".pt", "")

    return {
        "usuario": best_user,
        "score": round(best_score, 4)
    }
