# Multimodal AI Product Search Engine

## Quick Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Build search index:**
```bash
python build_index.py
```

3. **Start server:**
```bash
python main.py
```

4. **Open browser:** http://localhost:8000

## Files Required
- `main.py` - FastAPI server
- `build_index.py` - Product database & index builder
- `clip_encoder.py` - CLIP model wrapper
- `faiss_index.py` - Vector search implementation
- `index.html` - Web interface
- `requirements.txt` - Dependencies

## Features
- Text & image search
- 125+ products across 5 categories
- Shopping cart & favorites
- Price/category filters
- Mobile responsive design