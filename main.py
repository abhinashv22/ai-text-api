from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from transformers import pipeline
from functools import lru_cache
import logging

app = FastAPI(
    title="AI Text Intelligence API",
    description="Advanced NLP API with AI summarization, keyword extraction, similarity & language detection",
    version="2.1"
)

# Logging setup
logging.basicConfig(level=logging.INFO)

# 🔥 LAZY LOAD MODEL (CRITICAL FOR DEPLOYMENT)
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        logging.info("Loading AI model...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

# Request Models
class TextRequest(BaseModel):
    text: str

class CompareRequest(BaseModel):
    text1: str
    text2: str

# Home Endpoint
@app.get("/")
async def home():
    return {"message": "API is running 🚀"}

# Word Count
@app.post("/wordcount")
async def word_count(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    words = text.split()

    return {
        "total_words": len(words),
        "preview": text[:50],
        "status": "processed"
    }

# 🔥 AI Summarization (LAZY + CACHED)
@lru_cache(maxsize=10)
def generate_summary(text: str):
    model = get_summarizer()
    return model(text, max_length=60, min_length=20, do_sample=False)

@app.post("/summarize")
async def summarize(request: TextRequest):
    text = request.text.strip()

    if len(text) < 50:
        return {"warning": "Text too short for AI summarization"}

    try:
        result = generate_summary(text)
        return {
            "summary": result[0]["summary_text"]
        }
    except Exception as e:
        return {"error": "Summarization failed", "details": str(e)}

# 🔥 TF-IDF Keyword Extraction
@app.post("/keywords")
async def keywords(request: TextRequest):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    X = vectorizer.fit_transform([request.text])
    return {"keywords": vectorizer.get_feature_names_out().tolist()}

# Text Similarity
@app.post("/similarity")
async def similarity(req: CompareRequest):
    texts = [req.text1, req.text2]
    vectorizer = CountVectorizer().fit_transform(texts)
    similarity_score = cosine_similarity(vectorizer)[0][1]
    return {"similarity": float(similarity_score)}

# 🔥 Language Detection
@app.post("/language")
async def detect_language(request: TextRequest):
    text = request.text.strip()

    if len(text) < 20:
        return {"warning": "Text too short for accurate detection"}

    try:
        lang = detect(text)
        return {"language": lang}
    except:
        return {"error": "Language detection failed"}

# File Upload
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    return {"word_count": len(text.split())}