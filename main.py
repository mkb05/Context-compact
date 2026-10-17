from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from engine import VerbatimCompactorEngine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Context Compaction Service", version="1.0.0")

# Loaded once on startup (~65MB RAM)
compactor = VerbatimCompactorEngine(model_dir="./onnx_compactor")

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class CompactionRequest(BaseModel):
    raw_context: str = Field(..., description="Full multi-turn history to prune.")
    current_query: str = Field(..., description="Active user query guiding relevance.")
    target_reduction: float = Field(0.50, ge=0.1, le=0.9, description="Target reduction ratio.")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "context-compactor-onnx"}

@app.post("/api/v1/compact")
def compact_endpoint(req: CompactionRequest):
    try:
        return compactor.compact(
            raw_context=req.raw_context,
            current_query=req.current_query,
            target_reduction=req.target_reduction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))