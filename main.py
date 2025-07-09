from fastapi import FastAPI, UploadFile, File, Query
from pyngrok import ngrok
import torch
from torch.nn.functional import cosine_similarity
from pyannote.audio import Inference
import shutil
import os
import numpy as np

app = FastAPI()

# Cargar modelo de embeddings de voz
inference = Inference("pyannote/embedding", use_auth_token="TU_TOKEN_HF")

# Crear carpeta de embeddings si no existe
os.makedirs("embeddings", exist_ok=True)

# Rutas

@app.post("/register")
async def register_user(file: UploadFile = File(...), name: str = Query(...)):
    path = f"temp_register.wav"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = inference(path)
    os.remove(path)

    # Convertir a tensor si no lo es
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding.data, dtype=torch.float32)

    torch.save(embedding, f"embeddings/{name}.pt")
    return {"message": f"Usuario '{name}' registrado correctamente."}


@app.post("/identify")
async def identify_user(file: UploadFile = File(...)):
    path = "temp_identify.wav"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embedding = inference(path)
    os.remove(path)

    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding.data, dtype=torch.float32)

    best_score = -1.0
    best_user = "desconocido"

    for fname in os.listdir("embeddings"):
        ref_path = os.path.join("embeddings", fname)

        if os.path.isdir(ref_path):
            continue  # saltar carpetas como .ipynb_checkpoints

        ref = torch.load(ref_path)
        if not isinstance(ref, torch.Tensor):
            ref = torch.tensor(ref.data, dtype=torch.float32)

        score = cosine_similarity(embedding, ref, dim=0).item()
        if score > best_score:
            best_score = score
            best_user = fname.replace(".pt", "")

    return {
        "usuario": best_user,
        "score": round(best_score, 4)
    }

# Iniciar ngrok
public_url = ngrok.connect(8000, "http")
print(f"🌐 Tu servidor FastAPI está disponible públicamente en: {public_url}")

# Levantar FastAPI con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
