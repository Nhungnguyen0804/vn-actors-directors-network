"""
FastAPI Backend cho GraphRAG Chatbot
Cập nhật theo logic của model.py (với classify_intent + execute_query + format_result)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import time
import sys
from pathlib import Path

# Import các module GraphRAG
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.chatbot.entity_linking_node import (
        normalize_text,
        entity_linking_graph
    )
    from src.nlp.ner import extract_entity_from_sentences
    from src.chatbot.graph_query import close_driver
    from src.intent_pattern import INTENT_PATTERNS, FUNC_MAP

    # Import từ model.py
    from src.model_v2 import (
        load_llm_model,
        classify_intent,
        execute_query,
        format_result,
        llm_paraphrase
    )
    
    HAS_GRAPHRAG = True
except ImportError as e:
    print(f"Warning: GraphRAG modules not available - {e}")
    HAS_GRAPHRAG = False

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="GraphRAG Chatbot API",
    description="API cho hệ thống Q&A với Neo4j + LLM (Updated with model.py logic)",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class QuestionRequest(BaseModel):
    question: str
    debug: Optional[bool] = False
    use_finetuned: Optional[bool] = False

class StepOutput(BaseModel):
    id: str
    title: str
    type: str  # "llm", "neo4j", "logic"
    input: Any
    output: Any
    duration_ms: int

class ProgressInfo(BaseModel):
    status: str  # "processing", "done", "error"
    elapsed_ms: int
    estimated_total_ms: int

class ChatbotResponse(BaseModel):
    answer: str
    progress: ProgressInfo
    steps: List[StepOutput]

# ==================== GLOBAL STATE ====================

MODEL_PACK = None
MODEL_LOADING = False

def get_model():
    """Lazy load model với timeout"""
    global MODEL_PACK, MODEL_LOADING
    
    if MODEL_LOADING:
        print("Model is loading by another request, please wait...")
        return None
    
    if MODEL_PACK is None and HAS_GRAPHRAG:
        MODEL_LOADING = True
        print("Loading LLM model (this may take 30-60 seconds on first request)...")
        try:
            MODEL_PACK = load_llm_model(fine_tune=False)
            print("✅ Model loaded successfully and cached")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            MODEL_PACK = None
        finally:
            MODEL_LOADING = False
    
    return MODEL_PACK

# ==================== MAIN CHATBOT FUNCTION ====================

def chatbot_answer(question: str, debug: bool = False, use_finetuned: bool = False) -> Dict[str, Any]:
    """
    Main pipeline theo logic model.py:
    1. Extract entities
    2. Entity linking
    3. Classify intent (với num_entities)
    4. Execute query (graph hoặc custom function)
    5. Format result
    6. LLM paraphrase
    """
    
    start = time.time()
    steps: List[Dict[str, Any]] = []
    
    if not HAS_GRAPHRAG:
        return {
            "answer": "System not configured properly. GraphRAG modules missing.",
            "progress": {
                "status": "error",
                "elapsed_ms": 0,
                "estimated_total_ms": 0
            },
            "steps": []
        }
    
    try:
        # ===== STEP 1: EXTRACT ENTITIES =====
        t0 = time.time()
        entities = extract_entity_from_sentences(question)
        steps.append({
            "id": "extract_entities",
            "title": "Extract entities from question",
            "type": "llm",
            "input": question,
            "output": {
                "entities": entities if entities else [],
                "count": len(entities) if entities else 0
            },
            "duration_ms": int((time.time() - t0) * 1000)
        })
        
        if not entities:
            return {
                "answer": "Không tìm thấy tên riêng trong câu hỏi.",
                "progress": {
                    "status": "done",
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "estimated_total_ms": int((time.time() - start) * 1000)
                },
                "steps": steps
            }
        
        # ===== STEP 2: ENTITY LINKING =====
        t0 = time.time()
        linked_entities = entity_linking_graph(question)
        
        # Chuẩn hóa format (giống model.py)
        if linked_entities:
            normalized_entities = []
            for e in linked_entities:
                if isinstance(e, dict):
                    normalized_entities.append({
                        "node_name": e.get('node_name', e.get('name', 'Unknown')),
                        "node_id": e.get('node_id', e.get('node_name', 'Unknown')),
                        "type": e.get('type', 'person'),
                        "score": e.get('score', 0),
                        "match_type": e.get('match_type', 'unknown')
                    })
            linked_entities = normalized_entities
        
        steps.append({
            "id": "entity_linking",
            "title": "Link entities to Neo4j nodes",
            "type": "neo4j",
            "input": {
                "question": question,
                "entities": entities
            },
            "output": {
                "linked_entities": [
                    {
                        "node_name": e['node_name'],
                        "type": e.get('type', 'person'),
                        "score": e.get('score', 0),
                        "match_type": e.get('match_type', 'unknown')
                    } for e in (linked_entities or [])
                ],
                "count": len(linked_entities) if linked_entities else 0
            },
            "duration_ms": int((time.time() - t0) * 1000)
        })
        
        if not linked_entities:
            return {
                "answer": "Không tìm thấy thực thể phù hợp trong cơ sở dữ liệu.",
                "progress": {
                    "status": "done",
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "estimated_total_ms": int((time.time() - start) * 1000)
                },
                "steps": steps
            }
        
        # Extract entity info
        entity_name = linked_entities[0]['node_name']
        entity_type = linked_entities[0]['type'].upper()
        entity_name_2 = linked_entities[1]['node_name'] if len(linked_entities) > 1 else None
        num_entities = len(linked_entities)
        
        # ===== STEP 3: CLASSIFY INTENT (theo model.py) =====
        t0 = time.time()
        intent_name, config = classify_intent(
            question, 
            entity_type=entity_type,
            num_entities=num_entities,
            debug=debug
        )
        
        steps.append({
            "id": "classify_intent",
            "title": "Classify question intent",
            "type": "logic",
            "input": {
                "question": question,
                "entity_type": entity_type,
                "num_entities": num_entities
            },
            "output": {
                "intent": intent_name,
                "relationships": config.get("rels", []),
                "function": config.get("func"),
                "needs_2_entities": config.get("needs_2_entities", False),
                "max_entities": config.get("max_entities")
            },
            "duration_ms": int((time.time() - t0) * 1000)
        })
        
        # ===== STEP 4: EXECUTE QUERY (theo model.py) =====
        t0 = time.time()
        data = execute_query(
            entity_name=entity_name,
            entity_type=entity_type,
            intent_name=intent_name,
            config=config,
            entity_name_2=entity_name_2,
            debug=debug
        )
        
        query_type = "function" if "func" in config else "relationship"
        
        steps.append({
            "id": "execute_query",
            "title": "Execute Neo4j query",
            "type": "neo4j",
            "input": {
                "query_type": query_type,
                "entity_name": entity_name,
                "entity_name_2": entity_name_2,
                "relationships": config.get("rels", []),
                "function": config.get("func")
            },
            "output": {
                "status": "success" if data else "no_data",
                "data_type": type(data).__name__ if data else None,
                "data_preview": str(data)[:200] + "..." if data and len(str(data)) > 200 else str(data)
            },
            "duration_ms": int((time.time() - t0) * 1000)
        })
        
        if not data:
            return {
                "answer": "Không tìm thấy thông tin.",
                "progress": {
                    "status": "done",
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "estimated_total_ms": int((time.time() - start) * 1000)
                },
                "steps": steps
            }
        
        # ===== STEP 5: FORMAT RESULT (theo model.py) =====
        t0 = time.time()
        formatted = format_result(
            data=data,
            intent_name=intent_name,
            entity_name=entity_name,
            entity_name_2=entity_name_2
        )
        
        steps.append({
            "id": "format_result",
            "title": "Format graph data to text",
            "type": "logic",
            "input": {
                "data_type": type(data).__name__,
                "intent": intent_name
            },
            "output": {
                "formatted_text": formatted[:300] + "..." if len(formatted) > 300 else formatted
            },
            "duration_ms": int((time.time() - t0) * 1000)
        })
        
        # ===== STEP 6: LLM PARAPHRASE (theo model.py) =====
        t0 = time.time()
        model_pack = get_model()
        
        if model_pack is None:
            # Fallback: Clean formatted text
            final_answer = formatted
            final_answer = final_answer.replace("KHONG TIM THAY", "Không tìm thấy")
            final_answer = final_answer.replace("THONG TIN:", "Thông tin:")
            final_answer = final_answer.replace("DANH SACH:", "Danh sách:")
            final_answer = final_answer.replace("PHIM:", "Phim:")
            final_answer = final_answer.replace("THE LOAI:", "Thể loại:")
            final_answer = final_answer.replace("DAO DIEN:", "Đạo diễn:")
            final_answer = final_answer.replace("DIEN VIEN:", "Diễn viên:")
            final_answer = final_answer.replace("VO/CHONG:", "Vợ/chồng:")
            final_answer = final_answer.replace("NAM SINH:", "Năm sinh:")
            final_answer = final_answer.replace("QUE QUAN:", "Quê quán:")
            
            steps.append({
                "id": "llm_paraphrase",
                "title": "Format answer (no LLM paraphrase)",
                "type": "logic",
                "input": None,
                "output": {
                    "answer": final_answer,
                    "note": "Model not loaded, using formatted text"
                },
                "duration_ms": int((time.time() - t0) * 1000)
            })
        else:
            # Use LLM paraphrase (theo model.py)
            final_answer = llm_paraphrase(
                model_pack=model_pack,
                formatted=formatted,
                question=question,
                debug=debug
            )
            
            # Check if anti-hallucination kicked in
            is_direct_data = (final_answer == formatted)
            
            steps.append({
                "id": "llm_paraphrase",
                "title": "Generate natural language answer",
                "type": "llm",
                "input": {
                    "formatted_text": formatted[:100] + "...",
                    "question": question,
                    "temperature": 0.0,
                    "greedy_decoding": True
                },
                "output": {
                    "answer": final_answer,
                    "direct_data_returned": is_direct_data,
                    "hallucination_prevented": is_direct_data
                },
                "duration_ms": int((time.time() - t0) * 1000)
            })
        
        # ===== FINAL RESPONSE =====
        elapsed_ms = int((time.time() - start) * 1000)
        estimated_total_ms = sum(s["duration_ms"] for s in steps)
        
        return {
            "answer": final_answer,
            "progress": {
                "status": "done",
                "elapsed_ms": elapsed_ms,
                "estimated_total_ms": estimated_total_ms
            },
            "steps": steps
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc() if debug else str(e)
        
        elapsed_ms = int((time.time() - start) * 1000)
        steps.append({
            "id": "error",
            "title": "Error occurred",
            "type": "logic",
            "input": None,
            "output": {
                "error": str(e),
                "detail": error_detail if debug else None
            },
            "duration_ms": 0
        })
        
        return {
            "answer": f"Đã xảy ra lỗi: {str(e)}",
            "progress": {
                "status": "error",
                "elapsed_ms": elapsed_ms,
                "estimated_total_ms": elapsed_ms
            },
            "steps": steps
        }

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "GraphRAG Chatbot API is running (v3.0 - Updated with model.py logic)",
        "graphrag_available": HAS_GRAPHRAG,
        "model_loaded": MODEL_PACK is not None,
        "features": [
            "classify_intent with entity_type + num_entities",
            "execute_query with custom functions",
            "format_result with intent templates",
            "llm_paraphrase with anti-hallucination (temperature=0.0)",
            "Spouse queries with max_entities=1",
            "Multi-entity queries"
        ]
    }

@app.post("/api/chat", response_model=ChatbotResponse)
async def chat(request: QuestionRequest):
    """
    Main chatbot endpoint
    
    Example request:
    ```json
    {
        "question": "Vợ Trấn Thành đóng phim gì?",
        "debug": false,
        "use_finetuned": false
    }
    ```
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = chatbot_answer(
            request.question, 
            debug=request.debug,
            use_finetuned=request.use_finetuned
        )
        return result
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error in chat endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    if MODEL_PACK is not None:
        model_status = "loaded"
    elif MODEL_LOADING:
        model_status = "loading"
    else:
        model_status = "not_loaded"
    
    return {
        "status": "healthy",
        "graphrag_available": HAS_GRAPHRAG,
        "model_status": model_status,
        "model_info": {
            "loaded": MODEL_PACK is not None,
            "loading": MODEL_LOADING,
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "note": "First query will be slower if model not preloaded"
        },
        "features": {
            "anti_hallucination": True,
            "temperature_0": True,
            "greedy_decoding": True,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3
        },
        "timestamp": time.time()
    }

@app.post("/api/preload-model")
async def preload_model():
    """
    Preload model to reduce first query latency
    """
    global MODEL_LOADING
    
    if MODEL_PACK is not None:
        return {
            "status": "success",
            "message": "Model already loaded"
        }
    
    if MODEL_LOADING:
        return {
            "status": "loading",
            "message": "Model is currently being loaded by another process"
        }
    
    try:
        import threading
        
        def load_in_background():
            get_model()
        
        thread = threading.Thread(target=load_in_background)
        thread.start()
        
        return {
            "status": "loading",
            "message": "Model loading started in background. This may take 30-60 seconds.",
            "tip": "Check /api/health to see when model is ready"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/intents")
async def list_intents():
    """List all available intents"""
    if not HAS_GRAPHRAG:
        raise HTTPException(status_code=503, detail="GraphRAG not available")
    
    intents_info = []
    for intent_name, config in INTENT_PATTERNS.items():
        intents_info.append({
            "name": intent_name,
            "patterns": config.get("patterns", []),
            "relationships": config.get("rels", []),
            "function": config.get("func"),
            "needs_2_entities": config.get("needs_2_entities", False),
            "max_entities": config.get("max_entities")
        })
    
    return {
        "total": len(intents_info),
        "intents": intents_info
    }

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print("\n" + "="*60)
    print("🚀 GraphRAG Chatbot API Starting (v3.0)...")
    print("="*60)
    print("✨ Features:")
    print("  - Updated with model.py logic")
    print("  - classify_intent + execute_query + format_result")
    print("  - Anti-hallucination (temperature=0.0, greedy)")
    print("  - Spouse queries with max_entities=1")
    print("="*60)
    
    PRELOAD_MODEL = True
    
    if PRELOAD_MODEL and HAS_GRAPHRAG:
        import threading
        def load_in_background():
            print("⏳ Preloading model in background...")
            get_model()
            print("✅ Model preloaded and ready!")
        
        thread = threading.Thread(target=load_in_background, daemon=True)
        thread.start()
    else:
        print("💡 Model will load on first query (lazy loading)")
    
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if HAS_GRAPHRAG:
        try:
            close_driver()
            print("Neo4j driver closed")
        except Exception as e:
            print(f"Error closing Neo4j driver: {e}")

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("GRAPHRAG CHATBOT API v3.0")
    print("="*60)
    print(f"GraphRAG modules available: {HAS_GRAPHRAG}")
    print("\nStarting server...")
    print("API docs will be available at: http://localhost:8000/docs")
    print("Test endpoint: http://localhost:8000/")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        log_level="info"
    )