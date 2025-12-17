# 🚀 Setup Guide for New PC

## Prerequisites
1. **Python 3.8+** installed
2. **Git** (optional, for cloning)

## Step 1: Copy Project Files
Copy the entire `Koushik_project` folder to your new PC.

## Step 2: Open Command Prompt
```bash
cd path\to\Koushik_project
```

## Step 3: Create Virtual Environment
```bash
python -m venv venv
```

## Step 4: Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# You should see (venv) at the start of your command line
```

## Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 6: Build Search Index
```bash
python build_index.py
```

## Step 7: Start the API Server
```bash
python main.py
```

## Step 8: Open Web Interface
Open `index.html` in your browser or double-click it.

## ✅ Test the Setup
1. Try text search: "blue shirt"
2. Try image upload search
3. Check if suggestions work

## 🔧 Troubleshooting

**Error: "Python not found"**
- Install Python from python.org
- Add Python to PATH during installation

**Error: "Module not found"**
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

**Error: "CLIP model download fails"**
- Check internet connection
- The first run downloads ~500MB CLIP model

**Error: "Index not found"**
- Run: `python build_index.py`
- Wait for "Index Building Complete!" message

## 📁 Project Structure
```
Koushik_project/
├── main.py              # API server
├── build_index.py       # Index builder
├── clip_encoder.py      # CLIP model wrapper
├── faiss_index.py       # Vector search
├── index.html           # Web interface
├── requirements.txt     # Dependencies
└── data/
    └── index/           # Search index files
```

## 🎯 Quick Start Commands
```bash
# Setup (run once)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python build_index.py

# Daily use
venv\Scripts\activate
python main.py
# Then open index.html
```