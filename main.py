from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from faster_whisper import WhisperModel
import os
from datetime import datetime
from pathlib import Path
import config
import numpy as np
import wave
import asyncio
from concurrent.futures import ThreadPoolExecutor
import httpx
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

Path(config.OUTPUT_DIR).mkdir(exist_ok=True)

# GPU Detection
def get_device_config():
    """Detect and configure device (CPU/GPU) and compute type."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    
    if config.DEVICE == "auto":
        device = "cuda" if has_cuda else "cpu"
    else:
        device = config.DEVICE
    
    if config.COMPUTE_TYPE == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = config.COMPUTE_TYPE
    
    num_workers = 1 if device == "cuda" else 2
    
    logger.info(f"Using device: {device}, compute_type: {compute_type}")
    return device, compute_type, num_workers

device, compute_type, num_workers = get_device_config()
current_model_name = config.WHISPER_MODEL
model = WhisperModel(current_model_name, device=device, compute_type=compute_type, num_workers=num_workers)
executor = ThreadPoolExecutor(max_workers=1)

# Runtime parameters
runtime_params = {
    "model": config.WHISPER_MODEL,
    "beam_size": 5,
    "min_audio_length_sec": config.MIN_AUDIO_LENGTH_SEC,
    "chunk_duration_ms": config.CHUNK_DURATION_MS,
    "language": "de",
    "temperature": 0.0,
    "best_of": 5,
    "vad_threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500,
    "condition_on_previous_text": True
}

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/config")
async def get_config():
    return {
        "chunkDuration": runtime_params["chunk_duration_ms"],
        "sampleRate": config.SAMPLE_RATE,
        "model": runtime_params["model"],
        "beamSize": runtime_params["beam_size"],
        "minAudioLength": runtime_params["min_audio_length_sec"],
        "language": runtime_params["language"],
        "temperature": runtime_params["temperature"],
        "bestOf": runtime_params["best_of"],
        "vadThreshold": runtime_params["vad_threshold"],
        "minSpeechDuration": runtime_params["min_speech_duration_ms"],
        "minSilenceDuration": runtime_params["min_silence_duration_ms"],
        "conditionOnPreviousText": runtime_params["condition_on_previous_text"]
    }

@app.post("/config")
async def update_config(request: Request):
    global model, current_model_name
    data = await request.json()
    
    # Model update
    if "model" in data and data["model"] != current_model_name:
        try:
            current_model_name = data["model"]
            device, compute_type, num_workers = get_device_config()
            model = WhisperModel(current_model_name, device=device, compute_type=compute_type, num_workers=num_workers)
            runtime_params["model"] = current_model_name
            logger.info(f"Model changed to: {current_model_name}")
        except Exception as e:
            logger.error(f"Model change error: {e}")
            return JSONResponse({"status": "error", "message": f"Modell-Fehler: {str(e)}"})
    
    # Update other parameters
    param_mapping = {
        "beamSize": ("beam_size", int),
        "minAudioLength": ("min_audio_length_sec", float),
        "chunkDuration": ("chunk_duration_ms", int),
        "language": ("language", str),
        "temperature": ("temperature", float),
        "bestOf": ("best_of", int),
        "vadThreshold": ("vad_threshold", float),
        "minSpeechDuration": ("min_speech_duration_ms", int),
        "minSilenceDuration": ("min_silence_duration_ms", int),
        "conditionOnPreviousText": ("condition_on_previous_text", bool)
    }
    
    for key, (param_name, param_type) in param_mapping.items():
        if key in data:
            runtime_params[param_name] = param_type(data[key])
    
    return JSONResponse({"status": "success", "params": runtime_params})

def transcribe_audio(audio_data, temp_file_path, beam_size, is_final=False):
    """Transcribe audio data to text."""
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)
    
    segments, _ = model.transcribe(
        temp_file_path, 
        language=runtime_params["language"],
        beam_size=beam_size,
        best_of=runtime_params["best_of"],
        temperature=runtime_params["temperature"],
        vad_filter=True,
        vad_parameters=dict(
            threshold=runtime_params["vad_threshold"],
            min_speech_duration_ms=runtime_params["min_speech_duration_ms"],
            min_silence_duration_ms=runtime_params["min_silence_duration_ms"]
        ),
        condition_on_previous_text=runtime_params["condition_on_previous_text"]
    )
    text = " ".join([segment.text for segment in segments])
    
    if not is_final and text.endswith('.'):
        text = text[:-1]
    
    try:
        os.remove(temp_file_path)
    except Exception as e:
        logger.warning(f"Could not remove temp file {temp_file_path}: {e}")
    
    return text

@app.get("/files")
async def list_files():
    try:
        files = [f for f in os.listdir(config.OUTPUT_DIR) if f.endswith('.txt')]
        files.sort(reverse=True)
        return JSONResponse({"files": files})
    except Exception as e:
        return JSONResponse({"files": []})

@app.get("/files/{filename}")
async def get_file_content(filename: str):
    """Get content of a saved transcription file."""
    try:
        filepath = Path(config.OUTPUT_DIR) / filename
        if not filepath.exists() or not filename.endswith('.txt'):
            return JSONResponse({"status": "error", "message": "Datei nicht gefunden"}, status_code=404)
        
        # Security: Prevent path traversal
        if not filepath.resolve().parent == Path(config.OUTPUT_DIR).resolve():
            return JSONResponse({"status": "error", "message": "Ungültiger Pfad"}, status_code=403)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return JSONResponse({"status": "success", "content": content})
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a saved transcription file."""
    try:
        filepath = Path(config.OUTPUT_DIR) / filename
        if not filepath.exists() or not filename.endswith('.txt'):
            return JSONResponse({"status": "error", "message": "Datei nicht gefunden"}, status_code=404)
        
        # Security: Prevent path traversal
        if not filepath.resolve().parent == Path(config.OUTPUT_DIR).resolve():
            return JSONResponse({"status": "error", "message": "Ungültiger Pfad"}, status_code=403)
        
        os.remove(filepath)
        logger.info(f"Deleted file: {filename}")
        return JSONResponse({"status": "success", "message": "Datei gelöscht"})
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and transcribe an audio file with streaming."""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac')):
        return JSONResponse({"status": "error", "message": "Nur Audio-Dateien erlaubt (.wav, .mp3, .m4a, .ogg, .flac)"}, status_code=400)
    
    temp_file = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=Path(file.filename).suffix) as tf:
            temp_file = tf.name
            
            # Streaming upload with size limit
            CHUNK_SIZE = 1024 * 1024  # 1MB chunks
            total_size = 0
            max_size = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024 if config.MAX_UPLOAD_SIZE_MB > 0 else float('inf')
            
            while chunk := await file.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > max_size:
                    raise HTTPException(413, f"Datei zu groß (max {config.MAX_UPLOAD_SIZE_MB}MB)")
                tf.write(chunk)
        
        logger.info(f"Uploaded file: {file.filename} ({total_size / 1024 / 1024:.2f}MB)")
        
        # Transcribe
        segments, _ = model.transcribe(
            temp_file,
            language=runtime_params["language"],
            beam_size=runtime_params["beam_size"],
            best_of=runtime_params["best_of"],
            temperature=runtime_params["temperature"],
            vad_filter=True,
            condition_on_previous_text=runtime_params["condition_on_previous_text"]
        )
        
        text = " ".join([segment.text for segment in segments])
        
        return JSONResponse({"status": "success", "text": text})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse({"status": "error", "message": f"Fehler: {str(e)}"}, status_code=500)
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")

@app.post("/correct")
async def correct_text(request: Request):
    """Correct transcribed text using LLM."""
    data = await request.json()
    text = data.get("text", "")
    if not text.strip():
        return JSONResponse({"status": "error", "message": "Kein Text vorhanden"}, status_code=400)
    
    try:
        if config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                return JSONResponse({"status": "error", "message": "OpenAI API Key nicht konfiguriert"}, status_code=500)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                    json={
                        "model": config.OPENAI_MODEL,
                        "messages": [
                            {"role": "system", "content": "Du korrigierst transkribierte Texte. Gib NUR den korrigierten Text zurück, ohne Erklärungen."},
                            {"role": "user", "content": f"Korrigiere Interpunktion, Grammatik und Rechtschreibung:\n\n{text}"}
                        ]
                    }
                )
                response.raise_for_status()
                result = response.json()
                corrected = result["choices"][0]["message"]["content"].strip()
                return JSONResponse({"status": "success", "text": corrected})
        else:  # ollama
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{config.LLM_BASE_URL}/api/generate",
                    json={
                        "model": config.LLM_MODEL,
                        "prompt": f"Dies ist ein transkribierter Audio-Text, versuche den Sinn zu erkennen und korrigiere bei Bedarf die interpunktion, Grammatik und Rechtschreibung. Gib NUR den korrigierten Text zurück, ohne Erklärungen:\n\n{text}",
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                corrected = result.get("response", "").strip()
                return JSONResponse({"status": "success", "text": corrected})
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM HTTP error: {e}")
        return JSONResponse({"status": "error", "message": f"LLM-Fehler: {e.response.status_code}"}, status_code=500)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return JSONResponse({"status": "error", "message": f"LLM-Fehler: {str(e)}"}, status_code=500)

@app.post("/save")
async def save_transcription(request: Request):
    """Save transcription to file."""
    data = await request.json()
    text = data.get("text", "")
    suffix = data.get("suffix", "")
    
    if not text.strip():
        return JSONResponse({"status": "error", "message": "Kein Text vorhanden"}, status_code=400)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize suffix
        suffix = "".join(c for c in suffix if c.isalnum() or c in ('-', '_')).strip()
        suffix_part = f"_{suffix}" if suffix else ""
        filename = f"transcription_{timestamp}{suffix_part}.txt"
        filepath = Path(config.OUTPUT_DIR) / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        
        logger.info(f"Saved transcription: {filename}")
        return JSONResponse({"status": "success", "filename": str(filepath)})
    except Exception as e:
        logger.error(f"Save error: {e}")
        return JSONResponse({"status": "error", "message": "Fehler beim Speichern"}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription."""
    await websocket.accept()
    audio_buffer = np.array([], dtype=np.int16)
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                data = message["bytes"]
                chunk = np.frombuffer(data, dtype=np.int16)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                min_audio_length = int(runtime_params["min_audio_length_sec"] * config.SAMPLE_RATE)
                
                if len(audio_buffer) >= min_audio_length:
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as tf:
                        temp_file = tf.name
                    
                    audio_data = audio_buffer[:min_audio_length].tobytes()
                    
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(executor, transcribe_audio, audio_data, temp_file, runtime_params["beam_size"], False)
                    
                    audio_buffer = audio_buffer[min_audio_length:]
                    
                    if text.strip() and len(text.strip()) > 3:
                        await websocket.send_json({"text": text})
            
            elif "text" in message and message["text"] == "stop":
                # Transcribe remaining audio buffer
                if len(audio_buffer) > config.SAMPLE_RATE:  # At least 1 second
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as tf:
                        temp_file = tf.name
                    
                    audio_data = audio_buffer.tobytes()
                    
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(executor, transcribe_audio, audio_data, temp_file, runtime_params["beam_size"], True)
                    
                    if text.strip() and len(text.strip()) > 3:
                        await websocket.send_json({"text": text})
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
