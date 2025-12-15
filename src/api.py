"""
FastAPI Backend cho GraphRAG Chatbot
Tích hợp với hệ thống GraphRAG đã có
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
    
    # Import các hàm chính từ file gốc
    from src.model_v1 import (  # Thay 'your_main_file' bằng tên file thật
        load_llm_model,
        detect_intent,
        route_graph_query_dynamic,
        format_graph_data_dynamic,
        llm_paraphrase_graphrag
    )
    
    HAS_GRAPHRAG = True
except ImportError as e:
    print(f"Warning: GraphRAG modules not available - {e}")
    HAS_GRAPHRAG = False

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="GraphRAG Chatbot API",
    description="API cho hệ thống Q&A với Neo4j + LLM",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class QuestionRequest(BaseModel):
    question: str
    debug: Optional[bool] = False

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

def get_model():
    """Lazy load model"""
    global MODEL_PACK
    if MODEL_PACK is None and HAS_GRAPHRAG:
        print("Loading LLM model...")
        MODEL_PACK = load_llm_model(fine_tune=False)
    return MODEL_PACK

# ==================== MAIN CHATBOT FUNCTION ====================

def chatbot_answer(question: str, debug: bool = False) -> Dict[str, Any]:
    """
    CONTRACT BACKEND - Tích hợp thật với GraphRAG
    =================
    UI CHỈ DÙNG FILE NÀY
    """
    
    start = time.time()
    steps: List[Dict[str, Any]] = []
    
    if not HAS_GRAPHRAG:
        # Fallback nếu không có GraphRAG
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
        
        # ===== STEP 2: DETECT INTENT =====
        t0 = time.time()
        intent = detect_intent(question)
        steps.append({
            "id": "detect_intent",
            "title": "Detect question intent",
            "type": "logic",
            "input": question,
            "output": {
                "intent": intent['intent'],
                "confidence": intent['confidence']
            },
            "duration_ms": int((time.time() - t0) * 1000)
        })
        
        # ===== STEP 3: ENTITY LINKING =====
        t0 = time.time()
        linked_entities = entity_linking_graph(question)
        
        # FIX: Xử lý trường hợp entity_linking_graph trả về format khác
        if linked_entities:
            # Chuẩn hóa format nếu thiếu keys
            normalized_entities = []
            for e in linked_entities:
                if isinstance(e, dict):
                    normalized_entities.append({
                        "node_name": e.get('node_name', e.get('name', 'Unknown')),
                        "node_label": e.get('node_label', e.get('label', 'PERSON'))
                    })
                elif isinstance(e, str):
                    # Nếu chỉ là string, tạo dict mới
                    normalized_entities.append({
                        "node_name": e,
                        "node_label": "PERSON"  # Default
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
                        "node_label": e['node_label']
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
        
        # ===== STEP 4: QUERY NEO4J =====
        t0 = time.time()
        g_res = route_graph_query_dynamic(
            linked_entities, 
            question, 
            intent, 
            debug=debug
        )
        
        # Tạo Cypher query mô tả (để hiển thị)
        entity_name = g_res.get('entity_name', linked_entities[0]['node_name'])
        cypher_desc = f"MATCH (n)-[r]->(m) WHERE n.name = '{entity_name}' RETURN m"
        
        steps.append({
            "id": "query_neo4j",
            "title": "Query Neo4j graph database",
            "type": "neo4j",
            "input": {
                "cypher": cypher_desc,
                "intent": intent['intent'],
                "entities": [e['node_name'] for e in linked_entities]
            },
            "output": {
                "status": g_res['status'],
                "records_count": len(g_res['data']) if isinstance(g_res['data'], list) else 1,
                "data_preview": str(g_res['data'])[:200] + "..." if len(str(g_res['data'])) > 200 else str(g_res['data'])
            },
            "duration_ms": int((time.time() - t0) * 1000)
        })
        
        if g_res['status'] == 'error':
            return {
                "answer": g_res['message'],
                "progress": {
                    "status": "done",
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "estimated_total_ms": int((time.time() - start) * 1000)
                },
                "steps": steps
            }
        
        # ===== STEP 5: FORMAT DATA =====
        t0 = time.time()
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
        
        # ===== STEP 6: LLM PARAPHRASE =====
        t0 = time.time()
        model_pack = get_model()
        if model_pack is None:
            final_answer = formatted  # Fallback nếu không load được model
        else:
            final_answer = llm_paraphrase_graphrag(
                model_pack, 
                formatted, 
                question, 
                use_finetuned=False, 
                debug=debug
            )
        
        steps.append({
            "id": "llm_generate",
            "title": "Generate natural language answer",
            "type": "llm",
            "input": {
                "formatted_text": formatted[:100] + "...",
                "question": question
            },
            "output": {
                "answer": final_answer
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
        elapsed_ms = int((time.time() - start) * 1000)
        steps.append({
            "id": "error",
            "title": "Error occurred",
            "type": "logic",
            "input": None,
            "output": {"error": str(e)},
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
        "message": "GraphRAG Chatbot API is running",
        "graphrag_available": HAS_GRAPHRAG
    }

@app.post("/api/chat", response_model=ChatbotResponse)
async def chat(request: QuestionRequest):
    """
    Main chatbot endpoint
    
    Example request:
    ```json
    {
        "question": "Trấn Thành đóng phim gì?",
        "debug": false
    }
    ```
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = chatbot_answer(request.question, debug=request.debug)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if MODEL_PACK is not None else "not_loaded"
    
    return {
        "status": "healthy",
        "graphrag_available": HAS_GRAPHRAG,
        "model_status": model_status,
        "timestamp": time.time()
    }

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
    
    test_question = "Trấn Thành đóng phim gì?"
    result = chatbot_answer(test_question, debug=True)
    
    # Hiển thị JSON đẹp
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60 + "\n")
    exit()
    """
    
    # ===== SERVER MODE =====
    print("\n" + "="*60)
    print("GRAPHRAG CHATBOT API")
    print("="*60)
    print(f"GraphRAG modules available: {HAS_GRAPHRAG}")
    print("\nStarting server...")
    print("API docs will be available at: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )