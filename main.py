# requirements.txt
"""
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
numpy==1.24.3
librosa==0.10.1
sqlalchemy==2.0.23
aiosqlite==0.19.0
pyngrok==7.0.0
prometheus-client==0.19.0
python-multipart==0.0.6
"""

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from prometheus_client import Counter, Histogram, generate_latest
import numpy as np
import librosa
import os
import time
import asyncio
import logging
import pickle
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import hashlib
import uuid



# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración
EMBEDDINGS_DIR = "embeddings"
DATABASE_URL = "sqlite:///./voice_identification.db"
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CONFIDENCE_THRESHOLD = 0.7
SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 30  # segundos

# Métricas Prometheus
requests_total = Counter('requests_total', 'Total requests', ['endpoint', 'status'])
request_duration = Histogram('request_duration_seconds', 'Request duration', ['endpoint'])
identifications_total = Counter('identifications_total', 'Total identifications', ['result'])
registrations_total = Counter('registrations_total', 'Total registrations')

# Crear directorios
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Base de datos
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    embedding_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_identified = Column(DateTime)
    identification_count = Column(Integer, default=0)
    sample_count = Column(Integer, default=1)

class IdentificationLog(Base):
    __tablename__ = "identification_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    identified_user = Column(String)
    confidence_score = Column(Float)
    processing_time = Column(Float)
    file_hash = Column(String)

# Modelos Pydantic
class UserResponse(BaseModel):
    id: str
    name: str
    created_at: datetime.datetime
    last_identified: Optional[datetime.datetime]
    identification_count: int
    sample_count: int

class CandidateResponse(BaseModel):
    name: str
    score: float
    confidence: str

class IdentificationResponse(BaseModel):
    usuario: str
    score: float
    confianza: str
    candidatos: List[CandidateResponse]
    procesamiento_tiempo: float
    archivo_hash: str

class RegisterResponse(BaseModel):
    status: str
    msg: str
    user_id: str
    embedding_hash: str

class StatsResponse(BaseModel):
    total_users: int
    total_identifications: int
    avg_confidence: float
    top_users: List[Dict[str, Any]]

# Cache de embeddings
embeddings_cache: Dict[str, np.ndarray] = {}
cache_last_updated = 0

# Thread pool para procesamiento asíncrono
executor = ThreadPoolExecutor(max_workers=4)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def validate_audio_file(file: UploadFile) -> None:
    """Validar archivo de audio"""
    if not file.filename:
        raise HTTPException(400, "Nombre de archivo requerido")
    
    # Validar extensión
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, 
            f"Formato no válido. Extensiones permitidas: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Validar tamaño (esto es una aproximación, el tamaño real se valida después)
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(400, f"Archivo muy grande. Máximo: {MAX_FILE_SIZE//1024//1024}MB")

def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Preprocesar audio para mejorar calidad"""
    try:
        # Normalizar volumen
        y = librosa.util.normalize(y)
        
        # Aplicar preénfasis
        y = librosa.effects.preemphasis(y)
        
        # Detectar y extraer solo la voz (trim silence)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Validar duración
        duration = len(y_trimmed) / sr
        if duration > MAX_AUDIO_DURATION:
            # Truncar a duración máxima
            y_trimmed = y_trimmed[:MAX_AUDIO_DURATION * sr]
        
        if duration < 0.5:  # Mínimo 0.5 segundos
            raise ValueError("Audio muy corto")
            
        return y_trimmed
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {str(e)}")
        raise

def extract_embedding(path: str) -> np.ndarray:
    """Extraer TODAS las características disponibles en librosa"""
    try:
        # Cargar audio
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        
        # Preprocesar
        y = preprocess_audio(y, sr)

        # === CARACTERÍSTICAS ESPECTRALES ===
        # MFCC y variaciones
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # === CARACTERÍSTICAS TONALES ===
        # Chroma (todas las variantes)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        
        # Tonnetz (análisis armónico)
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        
        # === CARACTERÍSTICAS TEMPORALES ===
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        
        # === CARACTERÍSTICAS ESPECTRALES DETALLADAS ===
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        
        # Spectral Rolloff (percentiles)
        spectral_rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.1)
        spectral_rolloff_max = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.9)
        
        # === CARACTERÍSTICAS DE PITCH ===
        # YIN pitch tracking
        try:
            pitch_yin = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            pitch_yin = np.nan_to_num(pitch_yin, nan=0.0)
        except:
            pitch_yin = np.zeros(librosa.time_to_frames(len(y)/sr, sr=sr))
        
        # PYIN pitch tracking (más robusto)
        try:
            pitch_pyin = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            pitch_pyin = np.nan_to_num(pitch_pyin[0], nan=0.0)  # Solo la frecuencia
        except:
            pitch_pyin = np.zeros(librosa.time_to_frames(len(y)/sr, sr=sr))
        
        # === CARACTERÍSTICAS ARMÓNICAS/PERCUSIVAS ===
        # Separación HPSS
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Características de la parte armónica
        harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)
        harmonic_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr)
        harmonic_bandwidth = librosa.feature.spectral_bandwidth(y=y_harmonic, sr=sr)
        
        # Características de la parte percusiva
        percussive_centroid = librosa.feature.spectral_centroid(y=y_percussive, sr=sr)
        percussive_rolloff = librosa.feature.spectral_rolloff(y=y_percussive, sr=sr)
        
        # === CARACTERÍSTICAS DE TEMPO Y RITMO ===
        # Tempo y beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Onset frames
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        
        # === CARACTERÍSTICAS DE POLYPHONIC PITCH ===
        # Pitch class profile
        try:
            pcp = librosa.feature.pitch_class_profile(y=y, sr=sr)
        except:
            pcp = np.zeros((12, 1))
        
        # === CARACTERÍSTICAS DE COMPLEJIDAD ===
        # Spectral complexity
        try:
            spectral_complexity = librosa.feature.spectral_complexity(y=y, sr=sr)
        except:
            spectral_complexity = np.zeros((1, 1))
        
        # === ESTADÍSTICAS AVANZADAS ===
        def extract_comprehensive_stats(feature):
            """Extraer estadísticas completas de una característica"""
            if len(feature.shape) == 1:
                feature = feature.reshape(1, -1)
            
            stats = []
            for i in range(feature.shape[0]):
                row = feature[i]
                # Estadísticas básicas
                stats.extend([
                    np.mean(row), np.std(row), np.min(row), np.max(row),
                    np.median(row), np.percentile(row, 25), np.percentile(row, 75)
                ])
                # Estadísticas avanzadas
                stats.extend([
                    np.var(row),  # Varianza
                    np.sqrt(np.mean(row**2)),  # RMS
                    np.sum(np.abs(row)),  # Suma de valores absolutos
                    np.sum(row**2),  # Energía
                    np.max(row) - np.min(row),  # Rango
                    np.percentile(row, 10),  # Percentil 10
                    np.percentile(row, 90),  # Percentil 90
                    np.sum(row > np.mean(row)),  # Valores sobre la media
                    np.sum(row < np.mean(row))   # Valores bajo la media
                ])
            return np.array(stats)
        
        # === CONCATENAR TODAS LAS CARACTERÍSTICAS ===
        all_features = []
        
        # MFCC y deltas
        all_features.extend([
            extract_comprehensive_stats(mfcc),
            extract_comprehensive_stats(mfcc_delta),
            extract_comprehensive_stats(mfcc_delta2)
        ])
        
        # Mel spectrogram
        all_features.append(extract_comprehensive_stats(mel_db))
        
        # Chroma features
        all_features.extend([
            extract_comprehensive_stats(chroma_stft),
            extract_comprehensive_stats(chroma_cqt),
            extract_comprehensive_stats(chroma_cens)
        ])
        
        # Tonnetz
        all_features.append(extract_comprehensive_stats(tonnetz))
        
        # Características temporales
        all_features.extend([
            extract_comprehensive_stats(zcr),
            extract_comprehensive_stats(rms)
        ])
        
        # Características espectrales
        all_features.extend([
            extract_comprehensive_stats(spectral_centroid),
            extract_comprehensive_stats(spectral_rolloff),
            extract_comprehensive_stats(spectral_bandwidth),
            extract_comprehensive_stats(spectral_contrast),
            extract_comprehensive_stats(spectral_flatness),
            extract_comprehensive_stats(spectral_rolloff_min),
            extract_comprehensive_stats(spectral_rolloff_max)
        ])
        
        # Pitch features
        all_features.extend([
            extract_comprehensive_stats(pitch_yin),
            extract_comprehensive_stats(pitch_pyin)
        ])
        
        # Características armónicas
        all_features.extend([
            extract_comprehensive_stats(harmonic_centroid),
            extract_comprehensive_stats(harmonic_rolloff),
            extract_comprehensive_stats(harmonic_bandwidth),
            extract_comprehensive_stats(percussive_centroid),
            extract_comprehensive_stats(percussive_rolloff)
        ])
        
        # Características de tempo
        all_features.extend([
            np.array([tempo]),  # Tempo como escalar
            extract_comprehensive_stats(onset_env)
        ])
        
        # PCP
        all_features.append(extract_comprehensive_stats(pcp))
        
        # Spectral complexity
        all_features.append(extract_comprehensive_stats(spectral_complexity))
        
        # Concatenar todo
        features = np.concatenate(all_features)
        
        # Normalizar features
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
        
    except Exception as e:
        logger.error(f"Error extrayendo embedding: {str(e)}")
        raise

def calculate_similarity_advanced(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calcular similitud usando similitud coseno mejorada"""
    try:
        # Similitud coseno con manejo de errores
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Asegurar que esté en el rango [-1, 1] y convertir a [0, 1]
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        similarity = (cosine_sim + 1.0) / 2.0  # Convertir de [-1,1] a [0,1]
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error calculando similitud: {str(e)}")
        return 0.0

def calculate_confidence_level(score: float) -> str:
    """Calcular nivel de confianza"""
    if score >= 0.85:
        return "muy_alta"
    elif score >= 0.75:
        return "alta"
    elif score >= 0.65:
        return "media"
    elif score >= 0.5:
        return "baja"
    else:
        return "muy_baja"

def load_embeddings_cache():
    """Cargar embeddings en cache"""
    global embeddings_cache, cache_last_updated
    
    try:
        embeddings_cache.clear()
        for fname in os.listdir(EMBEDDINGS_DIR):
            if fname.endswith('.npy'):
                user_name = fname.replace('.npy', '')
                embedding_path = os.path.join(EMBEDDINGS_DIR, fname)
                embedding = np.load(embedding_path)
                embeddings_cache[user_name] = embedding
        
        cache_last_updated = time.time()
        logger.info(f"Cache actualizado con {len(embeddings_cache)} usuarios")
    except Exception as e:
        logger.error(f"Error cargando cache: {str(e)}")

def get_file_hash(content: bytes) -> str:
    """Obtener hash del archivo"""
    return hashlib.md5(content).hexdigest()

def save_embedding(name: str, embedding: np.ndarray) -> str:
    """Guardar embedding y retornar hash"""
    embedding_path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
    np.save(embedding_path, embedding)
    
    # Calcular hash del embedding
    embedding_bytes = embedding.tobytes()
    embedding_hash = hashlib.md5(embedding_bytes).hexdigest()
    
    # Actualizar cache
    embeddings_cache[name] = embedding
    
    return embedding_hash

def process_identification(file_content: bytes) -> Dict[str, Any]:
    """Procesar identificación en hilo separado"""
    start_time = time.time()
    
    # Crear archivo temporal
    temp_filename = f"temp_{int(time.time())}_{os.getpid()}.wav"
    
    try:
        # Escribir archivo temporal
        with open(temp_filename, "wb") as f:
            f.write(file_content)
        
        # Extraer embedding
        embedding = extract_embedding(temp_filename)
        
        # Buscar mejor coincidencia
        best_user = "desconocido"
        best_score = -1
        candidates = []
        
        for user_name, ref_embedding in embeddings_cache.items():
            # Similitud avanzada usando múltiples métricas
            score = calculate_similarity_advanced(embedding, ref_embedding)
            
            candidates.append({
                "name": user_name,
                "score": float(score),
                "confidence": calculate_confidence_level(score)
            })
            
            if score > best_score:
                best_score = score
                best_user = user_name
        
        # Ordenar candidatos por score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:3]  # Top 3
        
        # Aplicar umbral de confianza
        if best_score < CONFIDENCE_THRESHOLD:
            best_user = "desconocido"
        
        processing_time = time.time() - start_time
        file_hash = get_file_hash(file_content)
        
        return {
            "usuario": best_user,
            "score": round(best_score, 4),
            "confianza": calculate_confidence_level(best_score),
            "candidatos": top_candidates,
            "procesamiento_tiempo": round(processing_time, 3),
            "archivo_hash": file_hash
        }
    
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando aplicación...")
    
    # Crear tablas
    Base.metadata.create_all(bind=engine)
    
    # Cargar cache de embeddings
    load_embeddings_cache()
    
    # Configurar ngrok
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(8000)
        logger.info(f"🔗 URL pública: {public_url}")
    except Exception as e:
        logger.warning(f"No se pudo configurar ngrok: {e}")
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación...")
    executor.shutdown(wait=True)

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Identificación por Voz",
    description="API para registro e identificación de usuarios por voz",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de métricas
@app.middleware("http")
async def add_metrics(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    endpoint = request.url.path
    status = "success" if response.status_code < 400 else "error"
    
    requests_total.labels(endpoint=endpoint, status=status).inc()
    request_duration.labels(endpoint=endpoint).observe(duration)
    
    return response

@app.post("/register", response_model=RegisterResponse)
async def register_user(
    file: UploadFile = File(...), 
    name: str = Query(..., min_length=1, max_length=50),
    db: Session = Depends(get_db)
):
    """Registrar nuevo usuario con muestra de voz"""
    start_time = time.time()
    
    try:
        # Validaciones
        validate_audio_file(file)
        name = name.strip()
        
        # Verificar si usuario ya existe
        existing_user = db.query(User).filter(User.name == name).first()
        if existing_user:
            raise HTTPException(400, f"Usuario '{name}' ya existe")
        
        # Leer archivo
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(400, f"Archivo muy grande. Máximo: {MAX_FILE_SIZE//1024//1024}MB")
        
        # Procesar en hilo separado
        temp_filename = f"temp_reg_{int(time.time())}_{os.getpid()}.wav"
        
        try:
            with open(temp_filename, "wb") as f:
                f.write(file_content)
            
            # Extraer embedding
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(executor, extract_embedding, temp_filename)
            
            # Guardar embedding
            embedding_hash = save_embedding(name, embedding)
            
            # Crear usuario en BD
            user = User(
                name=name,
                embedding_hash=embedding_hash,
                sample_count=1
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Métricas
            registrations_total.inc()
            
            logger.info(f"Usuario '{name}' registrado exitosamente")
            
            return RegisterResponse(
                status="success",
                msg=f"Usuario '{name}' registrado exitosamente",
                user_id=user.id,
                embedding_hash=embedding_hash
            )
            
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registrando usuario '{name}': {str(e)}")
        raise HTTPException(500, f"Error procesando archivo: {str(e)}")

@app.post("/identify", response_model=IdentificationResponse)
async def identify_user(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Identificar usuario por voz"""
    try:
        # Validaciones
        validate_audio_file(file)
        
        # Verificar que hay usuarios registrados
        if not embeddings_cache:
            raise HTTPException(400, "No hay usuarios registrados")
        
        # Leer archivo
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(400, f"Archivo muy grande. Máximo: {MAX_FILE_SIZE//1024//1024}MB")
        
        # Procesar identificación
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_identification, file_content)
        
        # Actualizar estadísticas si se identificó usuario
        if result["usuario"] != "desconocido":
            user = db.query(User).filter(User.name == result["usuario"]).first()
            if user:
                user.last_identified = datetime.datetime.utcnow()
                user.identification_count += 1
                db.commit()
        
        # Registrar log
        log_entry = IdentificationLog(
            identified_user=result["usuario"],
            confidence_score=result["score"],
            processing_time=result["procesamiento_tiempo"],
            file_hash=result["archivo_hash"]
        )
        db.add(log_entry)
        db.commit()
        
        # Métricas
        result_type = "identified" if result["usuario"] != "desconocido" else "unknown"
        identifications_total.labels(result=result_type).inc()
        
        logger.info(f"Identificación: {result['usuario']} (score: {result['score']})")
        
        return IdentificationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en identificación: {str(e)}")
        raise HTTPException(500, f"Error procesando archivo: {str(e)}")

@app.get("/users", response_model=List[UserResponse])
async def get_users(db: Session = Depends(get_db)):
    """Obtener lista de usuarios registrados"""
    users = db.query(User).all()
    return users

@app.get("/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """Obtener estadísticas del sistema"""
    # Estadísticas básicas
    total_users = db.query(User).count()
    total_identifications = db.query(IdentificationLog).count()
    
    # Confianza promedio
    avg_confidence = db.query(IdentificationLog).filter(
        IdentificationLog.identified_user != "desconocido"
    ).with_entities(IdentificationLog.confidence_score).all()
    
    avg_conf_value = 0.0
    if avg_confidence:
        avg_conf_value = sum(score[0] for score in avg_confidence) / len(avg_confidence)
    
    # Top usuarios
    top_users = db.query(User).order_by(User.identification_count.desc()).limit(5).all()
    top_users_data = [
        {
            "name": user.name,
            "identifications": user.identification_count,
            "last_identified": user.last_identified.isoformat() if user.last_identified else None
        }
        for user in top_users
    ]
    
    return StatsResponse(
        total_users=total_users,
        total_identifications=total_identifications,
        avg_confidence=round(avg_conf_value, 4),
        top_users=top_users_data
    )

@app.delete("/users/{user_name}")
async def delete_user(user_name: str, db: Session = Depends(get_db)):
    """Eliminar usuario"""
    # Buscar usuario
    user = db.query(User).filter(User.name == user_name).first()
    if not user:
        raise HTTPException(404, f"Usuario '{user_name}' no encontrado")
    
    # Eliminar embedding
    embedding_path = os.path.join(EMBEDDINGS_DIR, f"{user_name}.npy")
    if os.path.exists(embedding_path):
        os.remove(embedding_path)
    
    # Eliminar del cache
    embeddings_cache.pop(user_name, None)
    
    # Eliminar de BD
    db.delete(user)
    db.commit()
    
    logger.info(f"Usuario '{user_name}' eliminado")
    
    return {"status": "success", "msg": f"Usuario '{user_name}' eliminado"}

@app.post("/reload-cache")
async def reload_cache():
    """Recargar cache de embeddings"""
    load_embeddings_cache()
    return {
        "status": "success", 
        "msg": f"Cache recargado con {len(embeddings_cache)} usuarios"
    }

@app.get("/metrics")
async def get_metrics():
    """Endpoint para métricas de Prometheus"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "users_in_cache": len(embeddings_cache),
        "cache_last_updated": cache_last_updated
    }

@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "name": "Sistema de Identificación por Voz",
        "version": "2.0.0",
        "status": "running",
        "users_registered": len(embeddings_cache),
        "endpoints": {
            "register": "POST /register - Registrar nuevo usuario",
            "identify": "POST /identify - Identificar usuario",
            "users": "GET /users - Listar usuarios",
            "stats": "GET /stats - Estadísticas",
            "health": "GET /health - Health check",
            "metrics": "GET /metrics - Métricas Prometheus"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="info"
    )