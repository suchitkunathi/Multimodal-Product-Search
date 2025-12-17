
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from clip_encoder import CLIPEncoder
from faiss_index import FAISSIndex


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Product Search API",
    description="AI-powered product search using images and text",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
encoder = None
index = None
INDEX_PATH = Path("data/index/products")


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global encoder, index
    
    print("\n" + "="*60)
    print("Starting Multimodal Product Search API")
    print("="*60)
    
    # Load CLIP encoder
    print("\n1. Loading CLIP encoder...")
    encoder = CLIPEncoder(model_name="ViT-B/32")
    
    # Load FAISS index
    print("\n2. Loading product index...")
    index = FAISSIndex(embedding_dim=512)
    
    if INDEX_PATH.with_suffix('.index').exists():
        index.load(str(INDEX_PATH))
        print(f"✓ Loaded index with {index.index.ntotal} products")
    else:
        print("⚠ Warning: No index found!")
        print(f"  Please run: python build_index.py")
        print(f"  Expected location: {INDEX_PATH}")
    
    print("\n" + "="*60)
    print("✓ API Ready!")
    print("="*60)
    print(f"Visit: http://localhost:8000/docs for API documentation")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """Health check endpoint"""
    stats = index.get_stats() if index else {'total_items': 0}
    return {
        "status": "online",
        "message": "Multimodal Product Search API",
        "total_products": stats['total_items'],
        "model": "CLIP ViT-B/32"
    }


@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    k: int = Form(10)
):
    """
    Search products by image.
    
    Args:
        file: Image file (PNG, JPG, JPEG)
        k: Number of results to return
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate embedding
        query_embedding = encoder.encode_image(image)
        
        # Search
        results = index.search(query_embedding, k=k)
        
        return {
            "query_type": "image",
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


@app.post("/search/text")
async def search_by_text(
    query: str = Form(...),
    k: int = Form(10)
):
    """
    Search products by text description.
    
    Args:
        query: Text description of desired product
        k: Number of results to return
    """
    try:
        if not query.strip():
            raise HTTPException(400, "Query cannot be empty")
        
        # Generate embedding
        query_embedding = encoder.encode_text(query)
        
        # Search
        results = index.search(query_embedding, k=k)
        
        return {
            "query_type": "text",
            "query": query,
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


@app.post("/search/hybrid")
async def hybrid_search(
    file: UploadFile = File(...),
    query: str = Form(...),
    alpha: float = Form(0.5),
    k: int = Form(10)
):
    """
    Hybrid search combining image and text.
    
    Args:
        file: Image file
        query: Text description
        alpha: Weight for image (0-1). Text weight = 1-alpha
        k: Number of results to return
    """
    try:
        # Validate inputs
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        if not query.strip():
            raise HTTPException(400, "Query cannot be empty")
        if not 0 <= alpha <= 1:
            raise HTTPException(400, "Alpha must be between 0 and 1")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate embeddings
        image_embedding = encoder.encode_image(image)
        text_embedding = encoder.encode_text(query)
        
        # Combine embeddings
        hybrid_embedding = alpha * image_embedding + (1 - alpha) * text_embedding
        
        # Normalize
        hybrid_embedding = hybrid_embedding / (hybrid_embedding ** 2).sum() ** 0.5
        
        # Search
        results = index.search(hybrid_embedding, k=k)
        
        return {
            "query_type": "hybrid",
            "text_query": query,
            "alpha": alpha,
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


# ============================================================================
# NEW: SEARCH FILTERS ENDPOINTS
# ============================================================================

@app.post("/search/filtered")
async def filtered_search(
    query: str = Form(None),
    file: Optional[UploadFile] = File(None),
    search_type: str = Form("text"),  # "text", "image", or "hybrid"
    min_price: float = Form(0),
    max_price: float = Form(100000),
    categories: str = Form(""),  # Comma-separated: "Clothing,Footwear"
    sort_by: str = Form("relevance"),  # "relevance", "price_low", "price_high"
    k: int = Form(50)  # Get more results before filtering
):
    """
    Advanced search with filters
    
    Filters:
    - Price range (min_price, max_price)
    - Categories (comma-separated list)
    - Sort by (relevance, price_low, price_high)
    """
    try:
        # Step 1: Get search results based on type
        if search_type == "image" and file:
            if not file.content_type.startswith('image/'):
                raise HTTPException(400, "File must be an image")
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            query_embedding = encoder.encode_image(image)
            
        elif search_type == "text" and query:
            if not query.strip():
                raise HTTPException(400, "Query cannot be empty")
            query_embedding = encoder.encode_text(query)
            
        elif search_type == "hybrid" and file and query:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_embedding = encoder.encode_image(image)
            text_embedding = encoder.encode_text(query)
            query_embedding = 0.5 * image_embedding + 0.5 * text_embedding
            query_embedding = query_embedding / (query_embedding ** 2).sum() ** 0.5
        else:
            raise HTTPException(400, "Invalid search type or missing parameters")
        
        # Step 2: Search
        results = index.search(query_embedding, k=k)
        
        # Step 3: Apply filters
        filtered_results = []
        
        # Parse categories
        category_list = [c.strip() for c in categories.split(",") if c.strip()]
        
        for result in results:
            # Price filter
            if result.get('price', 0) < min_price or result.get('price', 0) > max_price:
                continue
            
            # Category filter
            if category_list and result.get('category', '') not in category_list:
                continue
            
            filtered_results.append(result)
        
        # Step 4: Sort results
        if sort_by == "price_low":
            filtered_results.sort(key=lambda x: x.get('price', 0))
        elif sort_by == "price_high":
            filtered_results.sort(key=lambda x: x.get('price', 0), reverse=True)
        # else: keep relevance order (already sorted by similarity)
        
        # Return top 10
        final_results = filtered_results[:10]
        
        return {
            "query_type": search_type,
            "query": query if query else "image search",
            "filters_applied": {
                "price_range": [min_price, max_price],
                "categories": category_list,
                "sort_by": sort_by
            },
            "total_before_filter": len(results),
            "total_after_filter": len(filtered_results),
            "num_results": len(final_results),
            "results": final_results
        }
        
    except Exception as e:
        raise HTTPException(500, f"Filtered search failed: {str(e)}")


@app.get("/filters/categories")
async def get_categories():
    """Get list of all available categories"""
    if not index or not index.metadata:
        return {"categories": []}
    
    categories = set(item.get('category', 'Unknown') for item in index.metadata)
    return {"categories": sorted(list(categories))}


@app.get("/filters/price-range")
async def get_price_range():
    """Get min and max prices in catalog"""
    if not index or not index.metadata:
        return {"min": 0, "max": 0}
    
    prices = [item.get('price', 0) for item in index.metadata]
    return {
        "min": min(prices) if prices else 0,
        "max": max(prices) if prices else 0
    }


@app.get("/stats")
async def get_stats():
    """Get API and index statistics"""
    stats = index.get_stats() if index else {}
    return {
        "index_stats": stats,
        "model_info": {
            "clip_model": "ViT-B/32",
            "embedding_dim": encoder.get_embedding_dim() if encoder else None
        }
    }


@app.get("/search/suggestions")
async def get_search_suggestions(q: str = ""):
    """Get search suggestions based on query"""
    if not index or not index.metadata:
        return {"suggestions": []}
    
    popular_terms = [
        "blue shirt", "black shoes", "leather jacket", "running shoes", "denim jeans",
        "white sneakers", "brown wallet", "black hoodie", "red dress", "gray sweatshirt",
        "leather boots", "cotton t-shirt", "baseball cap", "crossbody bag", "wireless earbuds"
    ]
    
    suggestions = []
    query_lower = q.lower().strip()
    
    if len(query_lower) >= 2:
        matching_popular = [term for term in popular_terms if query_lower in term.lower()]
        suggestions.extend(matching_popular[:8])
    else:
        suggestions = popular_terms[:8]
    
    return {"suggestions": suggestions}


@app.get("/similar/{product_id}")
async def get_similar_products(product_id: str, k: int = 5):
    """Get similar products based on product ID"""
    if not index or not index.metadata:
        return {"similar": []}
    
    try:
        # Find the product
        product = None
        product_idx = None
        for i, item in enumerate(index.metadata):
            if str(item.get('id', i)) == str(product_id):
                product = item
                product_idx = i
                break
        
        if not product:
            return {"similar": []}
        
        # Re-encode the product description
        text = f"{product['name']} {product.get('category', '')} {product.get('description', '')}"
        product_embedding = encoder.encode_text(text)
        
        # Search for similar products
        results = index.search(product_embedding, k=k+1)  # +1 to exclude self
        
        # Filter out the original product
        similar = [r for r in results if str(r.get('id', '')) != str(product_id)][:k]
        
        return {"similar": similar}
        
    except Exception as e:
        return {"similar": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")