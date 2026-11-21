from __future__ import annotations
import os, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.getenv("MODEL_DIR", "../models/bert-tiny-e4")).expanduser()
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2000"))
LANGUAGE_NAMES = {"as-IN":"Assamese","bn-IN":"Bengali","en-IN":"English","gu-IN":"Gujarati","hi-IN":"Hindi","kn-IN":"Kannada","ml-IN":"Malayalam","mr-IN":"Marathi","pa-IN":"Punjabi","ta-IN":"Tamil","te-IN":"Telugu","none":"Other / unsupported"}
tokenizer = None
model = None
inference_lock = threading.Lock()

class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)

def load_model():
    global tokenizer, model
    if not MODEL_DIR.is_dir() or not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError("Model directory missing or has no config.json: " + str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model.to(torch.device("cpu"))
    model.eval()

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
        print("Loaded classifier from " + str(MODEL_DIR) + " on CPU")
    except Exception as exc:
        print("Classifier is not ready: " + str(exc))
    yield

app = FastAPI(title="Multilingual Language Classifier API", version="1.0.0", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins or ["*"], allow_credentials=False, allow_methods=["GET","POST","OPTIONS"], allow_headers=["Content-Type"])

def label_for(index: int) -> str:
    mapping: dict[Any, Any] = model.config.id2label or {}
    return str(mapping.get(index, mapping.get(str(index), "class-" + str(index))))

@app.get("/api/health")
def health():
    return {"status":"ok" if model is not None and tokenizer is not None else "model_not_loaded","model_loaded":model is not None and tokenizer is not None,"device":"cpu","model_dir_configured":str(MODEL_DIR)}

@app.get("/api/languages")
def languages():
    if model is None:
        raise HTTPException(status_code=503, detail="Classifier model is not loaded.")
    return {"languages":[{"code":label_for(i),"name":LANGUAGE_NAMES.get(label_for(i),label_for(i))} for i in range(int(model.config.num_labels))]}

@app.post("/api/classify")
def classify(payload: ClassifyRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Classifier model is not loaded. Check MODEL_DIR and backend logs.")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter a non-empty sentence.")
    started = time.perf_counter()
    try:
        with inference_lock, torch.inference_mode():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=int(getattr(model.config,"max_position_embeddings",192)))
            probabilities = torch.softmax(model(**inputs).logits[0], dim=-1)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed: " + str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    ranked = [{"code":label_for(i),"language":LANGUAGE_NAMES.get(label_for(i),label_for(i)),"score":round(float(p)*100,2)} for i,p in enumerate(probabilities.tolist())]
    ranked.sort(key=lambda item:item["score"], reverse=True)
    return {"input":text,"prediction":ranked[0],"alternatives":ranked[:5],"inference_ms":round(latency_ms,2),"device":"cpu","note":"Score is the model's softmax output, not a calibrated probability."}
