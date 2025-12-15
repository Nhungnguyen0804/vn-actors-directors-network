"""
FastAPI Backend cho GraphRAG Chatbot
Tích hợp với hệ thống GraphRAG đã có (với anti-hallucination)
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

    from src.model_v1 import ( 
        load_llm_model,
        detect_intent,
        route_graph_query_dynamic,
        format_graph_data_dynamic,
        llm_paraphrase_graphrag,
        compose_multi_queries,
        format_composed_results
    )
    
    HAS_GRAPHRAG = True
except ImportError as e:
    print(f"Warning: GraphRAG modules not available - {e}")
    HAS_GRAPHRAG = False

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="GraphRAG Chatbot API",
    description="API cho hệ thống Q&A với Neo4j + LLM (Anti-hallucination enabled)",
    version="2.0.0"
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
    CONTRACT BACKEND - Tích hợp thật với GraphRAG (with anti-hallucination)
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
        
        # Chuẩn hóa format
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
                elif isinstance(e, str):
                    normalized_entities.append({
                        "node_name": e,
                        "node_id": e,
                        "type": "person",
                        "score": 100,
                        "match_type": "direct"
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
        
        # ===== STEP 3: CHECK MULTI-QUERY =====
        t0 = time.time()
        num_entities = len(linked_entities)
        composed = None
        
        # Try multi-query composer
        try:
            composed = compose_multi_queries(question, linked_entities, debug=debug)
            if composed:
                steps.append({
                    "id": "multi_query_detection",
                    "title": "Multi-query detected and processed",
                    "type": "logic",
                    "input": {
                        "question": question,
                        "num_queries": len(composed)
                    },
                    "output": {
                        "queries_detected": list(composed.keys()),
                        "results": composed
                    },
                    "duration_ms": int((time.time() - t0) * 1000)
                })
        except Exception as e:
            if debug:
                print(f"[WARNING] Multi-query composer failed: {e}")
        
        # ===== STEP 4: DETECT INTENT =====
        if not composed:
            t0 = time.time()
            intent = detect_intent(question, num_entities=num_entities)
            
            # Log if 2+ entities forced to 2-hop query
            forced_2hop = False
            if num_entities >= 2 and intent['intent'] not in ['get_common_movies', 'get_common_directors', 'get_collaboration_history']:
                forced_2hop = True
            
            steps.append({
                "id": "detect_intent",
                "title": "Detect question intent",
                "type": "logic",
                "input": {
                    "question": question,
                    "num_entities": num_entities
                },
                "output": {
                    "intent": intent['intent'],
                    "confidence": intent['confidence'],
                    "entity_count": num_entities,
                    "forced_2hop": forced_2hop,
                    "note": "2+ entities requires 2-hop+ query" if forced_2hop else None
                },
                "duration_ms": int((time.time() - t0) * 1000)
            })
        
        # ===== STEP 5: QUERY NEO4J =====
        if composed:
            # Multi-query case
            entity_name = linked_entities[0]['node_name']
            formatted = format_composed_results(composed, entity_name)
            
            steps.append({
                "id": "query_neo4j",
                "title": "Execute multi-query on Neo4j",
                "type": "neo4j",
                "input": {
                    "type": "multi_query",
                    "entity": entity_name,
                    "sub_queries": list(composed.keys())
                },
                "output": {
                    "status": "success",
                    "results": composed,
                    "formatted": formatted
                },
                "duration_ms": 0  # Already counted in multi-query detection
            })
            
            g_res = {
                'status': 'success',
                'data': composed,
                'entity_name': entity_name,
                'formatted': formatted
            }
        else:
            # Single query case
            t0 = time.time()
            g_res = route_graph_query_dynamic(
                linked_entities, 
                question, 
                intent, 
                debug=debug
            )
            
            entity_name = g_res.get('entity_name', linked_entities[0]['node_name'])
            
            # Generate readable query description
            query_desc = f"Query for '{entity_name}' with intent '{intent['intent']}'"
            
            steps.append({
                "id": "query_neo4j",
                "title": "Query Neo4j graph database",
                "type": "neo4j",
                "input": {
                    "query_description": query_desc,
                    "intent": intent['intent'],
                    "entities": [e['node_name'] for e in linked_entities],
                    "num_entities": num_entities
                },
                "output": {
                    "status": g_res['status'],
                    "records_count": len(g_res['data']) if isinstance(g_res['data'], list) else 1,
                    "data_preview": str(g_res['data'])[:200] + "..." if len(str(g_res['data'])) > 200 else str(g_res['data']),
                    "has_fallback": 'formatted' in g_res
                },
                "duration_ms": int((time.time() - t0) * 1000)
            })
        
        if g_res['status'] == 'error':
            return {
                "answer": g_res.get('message', 'Đã xảy ra lỗi khi truy vấn dữ liệu.'),
                "progress": {
                    "status": "done",
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "estimated_total_ms": int((time.time() - start) * 1000)
                },
                "steps": steps
            }
        
        # ===== STEP 6: FORMAT DATA =====
        if not composed:
            t0 = time.time()
            if 'formatted' in g_res:
                formatted = g_res['formatted']
            else:
                formatted = format_graph_data_dynamic(
                    g_res['data'], 
                    intent['intent'], 
                    g_res.get('entity_name')
                )
            
            steps.append({
                "id": "format_data",
                "title": "Format graph data to text",
                "type": "logic",
                "input": {
                    "data_type": type(g_res['data']).__name__,
                    "intent": intent['intent']
                },
                "output": {
                    "formatted_text": formatted[:200] + "..." if len(formatted) > 200 else formatted
                },
                "duration_ms": int((time.time() - t0) * 1000)
            })
        
        # ===== STEP 7: LLM PARAPHRASE (with anti-hallucination) =====
        t0 = time.time()
        model_pack = get_model()
        
        if model_pack is None:
            # FALLBACK: Không cần model, trả về formatted text trực tiếp
            final_answer = formatted
            
            # Làm sạch formatted text để dễ đọc hơn
            final_answer = final_answer.replace("THONG TIN: ", "")
            final_answer = final_answer.replace("DANH SACH", "Danh sách")
            final_answer = final_answer.replace("PHIM:", "Các phim:")
            final_answer = final_answer.replace("THE LOAI:", "Thể loại:")
            final_answer = final_answer.replace("DAO DIEN:", "Đạo diễn:")
            final_answer = final_answer.replace("DIEN VIEN:", "Diễn viên:")
            final_answer = final_answer.replace("KHONG TIM THAY", "Không tìm thấy")
            
            steps.append({
                "id": "llm_generate",
                "title": "Format answer (no model paraphrase)",
                "type": "logic",
                "input": None,
                "output": {
                    "answer": final_answer,
                    "note": "Using formatted text without LLM paraphrase (faster response)",
                    "anti_hallucination": "N/A (no LLM used)"
                },
                "duration_ms": int((time.time() - t0) * 1000)
            })
        else:
            # Use LLM with anti-hallucination checks
            final_answer = llm_paraphrase_graphrag(
                model_pack, 
                formatted, 
                question, 
                use_finetuned=use_finetuned, 
                debug=debug
            )
            
            # Detect if anti-hallucination kicked in
            is_direct_data = (final_answer == formatted or 
                            final_answer in formatted or
                            formatted.split(":")[-1].strip().rstrip('.') == final_answer)
            
            steps.append({
                "id": "llm_generate",
                "title": "Generate natural language answer",
                "type": "llm",
                "input": {
                    "formatted_text": formatted[:100] + "...",
                    "question": question,
                    "anti_hallucination_enabled": True
                },
                "output": {
                    "answer": final_answer,
                    "direct_data_returned": is_direct_data,
                    "note": "Anti-hallucination checks passed" if not is_direct_data else "Returned direct data (hallucination prevented)"
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
        "message": "GraphRAG Chatbot API is running (v2.0 - Anti-hallucination enabled)",
        "graphrag_available": HAS_GRAPHRAG,
        "model_loaded": MODEL_PACK is not None,
        "features": [
            "Multi-query support",
            "Film genre queries",
            "Anti-hallucination checks",
            "2-entity = 2-hop enforcement"
        ]
    }

@app.post("/api/chat", response_model=ChatbotResponse)
async def chat(request: QuestionRequest):
    """
    Main chatbot endpoint
    
    Example request:
    ```json
    {
        "question": "Phim Nhà Bà Nữ thuộc thể loại gì?",
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
            "note": "First query will be slower if model not preloaded"
        },
        "features": {
            "anti_hallucination": True,
            "film_genre_query": True,
            "multi_query": True,
            "two_entity_enforcement": True
        },
        "timestamp": time.time()
    }

@app.post("/api/preload-model")
async def preload_model():
    """
    Preload model to reduce first query latency
    Call this endpoint after server starts to warm up the model
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
        
        # Load model in background thread
        thread = threading.Thread(target=load_in_background)
        thread.start()
        
        return {
            "status": "loading",
            "message": "Model loading started in background. This may take 30-60 seconds.",
            "tip": "Check /api/health to see when model is ready"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """
    Startup event - Preload model in background if needed
    """
    print("\n" + "="*60)
    print("🚀 GraphRAG Chatbot API Starting (v2.0)...")
    print("="*60)
    print("✨ Features:")
    print("  - Anti-hallucination checks")
    print("  - Film genre queries")
    print("  - Multi-query support")
    print("  - 2-entity = 2-hop enforcement")
    print("="*60)
    
    # Option 1: Preload model on startup (slower startup, faster first query)
    PRELOAD_MODEL = True  # Set to True to preload
    
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
    import json
    
    # ===== TEST MODE =====
    # Uncomment để test JSON output trước khi chạy server
    """
    print("\n" + "="*60)
    print("TEST JSON OUTPUT")
    print("="*60 + "\n")
    
    test_questions = [
        "Phim Nhà Bà Nữ thuộc thể loại gì?",
        "Trấn Thành đóng phim gì?",
        "Vợ của Trấn Thành sinh năm nào?",
        "Trấn Thành và Hari Won đóng chung phim nào?",
    ]
    
    for q in test_questions:
        print(f"\nQuestion: {q}")
        result = chatbot_answer(q, debug=True)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n" + "-"*60 + "\n")
    
    exit()
    """
    
    # ===== SERVER MODE =====
    print("\n" + "="*60)
    print("GRAPHRAG CHATBOT API v2.0")
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