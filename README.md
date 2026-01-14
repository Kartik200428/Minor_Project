# Multi-Agent Framework for Spam and Phishing Analysis

A comprehensive AI-powered system for analyzing and classifying emails as spam or phishing using a multi-agent LangGraph pipeline, machine learning classifiers, threat intelligence feeds, and local LLMs (Ollama).

## 🎯 Overview

This project implements an intelligent email analysis system that combines:
- **Machine Learning Classification**: TF-IDF + Logistic Regression for spam/phishing detection
- **Multi-Agent LangGraph Pipeline**: Orchestrates analysis through specialized nodes (ingest, filter, threat intel, explainability, response, SOC, forensics)
- **Threat Intelligence Integration**: Real-time URL reputation checks via OpenPhish and URLHaus
- **Local LLM Integration**: Uses Ollama for explainability, user guidance, and safe reply generation
- **Heuristic Scoring**: Rule-based features complement ML predictions

## ✨ Features

### Core Capabilities
- **Email Analysis**: Analyze individual emails with detailed risk scoring and explanations
- **Spam Classification**: Pre-trained SVC model for spam/ham classification
- **Phishing Detection**: Multi-stage pipeline combining ML, heuristics, and threat intelligence
- **Threat Intelligence**: Automatic URL reputation checking against known malicious domains
- **LLM-Powered Explanations**: Human-readable explanations of classification decisions
- **User Guidance**: Actionable recommendations for end users
- **SOC Recommendations**: Security operations center recommendations for analysts
- **Forensic Notes**: Detailed investigation logs for security teams


## 🏗️ Architecture

### System Components

```
┌─────────────────┐
│  Streamlit UI   │  ← User Interface
└────────┬────────┘
         │ HTTP/REST
┌────────▼────────┐
│  FastAPI Backend│  ← API Layer
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────────┐
│LangGraph│ │ ML Models       │
│Pipeline │ │                 │
└───┬───┘ └─────────────┘
    │
┌───▼────────────────────┐
│  Analysis Nodes:        │
│  • Ingest               │
│  • Filter (ML+Heur)     │
│  • Threat Intel         │
│  • Explainability (LLM) │
│  • Response (LLM)       │
│  • SOC                  │
│  • Forensics            │
└─────────────────────────┘
```

### LangGraph Pipeline Flow

1. **Ingest Node**: Normalizes input, combines subject + body
2. **Filter Node**: ML classification + heuristic scoring
3. **Threat Intel Node**: URL reputation checks (OpenPhish/URLHaus)
4. **Explainability Node**: LLM-generated explanations
5. **Response Node**: Safe reply suggestions and user guidance
6. **SOC Node**: Security operations recommendations
7. **Forensics Node**: Structured investigation notes

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- At least one Ollama model pulled (e.g., `qwen2.5:3b`, `qwen2.5:7b`, `llama3.1:8b`)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Agent-AI-System
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

1. Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull required models:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull qwen2.5:7b
   ollama pull llama3.1:8b
   ```

### 5. Prepare Training Data (Optional)

If you want to train your own models:

```bash
cd training
python prepare_dataset.py  # Combine raw datasets
python train_classifier.py  # Train the phishing classifier
Python Spam_classifier.py  # Train the spam classifier
```

## 🎮 Usage

### Starting the Backend API

```bash
# From project root
cd backend/api
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`

### Starting the Frontend

```bash
# From project root
streamlit run frontend/app.py
```

The UI will open in your browser at `http://localhost:8501`

### Testing the Pipeline

```bash
# Run the end-to-end test
python test_graph.py
```

## 📁 Project Structure

```
Agent-AI-System/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI application and endpoints
│   ├── core/
│   │   ├── config.py            # Global configuration
│   │   ├── model_loader.py      # ML model loading and inference
│   │   ├── llm_manager.py       # Ollama LLM integration
│   │   ├── ti_manager.py        # Threat intelligence feeds
│   │   ├── email_utils.py       # Email feature extraction
│   │   ├── eval_utils.py        # Offline evaluation utilities
│   │   ├── dataset_utils.py     # Dataset analysis tools
│   │   └── robustness_utils.py   # Adversarial testing
│   ├── graph/
│   │   ├── graph_builder.py     # LangGraph construction
│   │   ├── nodes.py             # Pipeline node implementations
│   │   └── state.py             # State type definitions
│   └── models/
│       └── email_classifier.joblib  # Trained phishing classifier
├── frontend/
│   └── app.py                   # Streamlit UI
├── training/
│   ├── prepare_dataset.py       # Dataset preprocessing
│   └── train_classifier.py      # Model training script
├── data/
│   ├── raw/                     # Raw datasets (CSV files)
│   ├── processed/               # Processed JSONL datasets
│   └── ti/                      # Threat intelligence cache
├── test_graph.py                # End-to-end pipeline test
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔌 API Endpoints

### Health & Configuration
- `GET /health` - Health check endpoint
- `GET /models` - List available LLM models

### Email Analysis
- `POST /analyze_email` - Analyze a single email through the full pipeline
- `POST /classify_spam` - Classify email as spam/ham using SVC model


### Example API Request

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/analyze_email",
    json={
        "subject": "Urgent: Verify your account",
        "body": "Please click here to verify: https://suspicious-site.com",
        "llm_model_name": "qwen2.5:3b"  # Optional
    }
)

result = response.json()
print(f"Decision: {result['decision']}")
print(f"Risk Score: {result['risk_score']}")
print(f"Explanation: {result['explanation']}")
```

### Model Architecture

- **Vectorizer**: TF-IDF with max_features=60000, ngram_range=(1,2)
- **Classifier**: Logistic Regression with balanced class weights
- **Features**: Combined subject + body text
- **Labels**: `phishing` or `benign` 'SPAM' or 'HAM'

## ⚙️ Configuration

Configuration is centralized in `backend/core/config.py`:

- **LLM Models**: Available models and default selection
- **Heuristic Weights**: ML, heuristic, and TI score blending
- **Thresholds**: Spam and Phishing detection and high-risk thresholds
- **Paths**: Data directories, model paths, TI cache locations
- **Threat Intelligence**: OpenPhish and URLHaus feed URLs

### Example Configuration

```python
# LLM Configuration
llm_config.available_models = ["qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"]
llm_config.default_model = "qwen2.5:3b"
llm_config.base_url = "http://localhost:11434"

# Heuristic Scoring Weights
heuristic_config.ml_weight = 0.6
heuristic_config.heuristic_weight = 0.2
heuristic_config.ti_weight = 0.2
```

## 🧩 Technologies Used

### Core Framework
- **FastAPI**: High-performance API framework
- **Streamlit**: Interactive web UI
- **LangGraph**: Multi-agent orchestration
- **LangChain**: LLM integration framework

### Machine Learning
- **scikit-learn**: Classification models (Logistic Regression, SVC)
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **joblib**: Model serialization

### LLM Integration
- **Ollama**: Local LLM server
- **langchain-ollama**: Ollama integration for LangChain

### Utilities
- **tldextract**: URL parsing and domain extraction
- **requests**: HTTP client for TI feeds

## 🔒 Security Considerations

- **Local LLM Processing**: All LLM inference runs locally via Ollama (no external API calls)
- **Threat Intelligence**: URL reputation checks use public feeds (OpenPhish, URLHaus)
- **Model Artifacts**: Trained models are stored locally and not exposed via API
- **CORS**: Currently configured for local development (adjust for production)



## 📝 Author

Kartik Singh
---



