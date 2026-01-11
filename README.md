# Multi-Agent Framework for Spam and Phishing Analysis

A comprehensive AI-powered system for analyzing and classifying emails as spam or phishing using a multi-agent LangGraph pipeline, machine learning classifiers, threat intelligence feeds, and local LLMs (Ollama).

## 🎯 Overview

This project implements an intelligent email analysis system that combines:
- **Machine Learning Classification**: TF-IDF + Logistic Regression for spam/phishing detection
- **Multi-Agent LangGraph Pipeline**: Orchestrates analysis through specialized nodes (ingest, filter, threat intel, explainability, response, SOC, forensics)
- **Threat Intelligence Integration**: Real-time URL reputation checks via OpenPhish and URLHaus
- **Local LLM Integration**: Uses Ollama for explainability, user guidance, and safe reply generation
- **Heuristic Scoring**: Rule-based features complement ML predictions
- **Evaluation Dashboard**: Comprehensive offline evaluation with ROC/PR curves, threshold analysis, and robustness testing

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

### Evaluation & Analytics
- **Offline Evaluation**: Comprehensive metrics on processed datasets
- **ROC & PR Curves**: Performance visualization for classifier quality assessment
- **Threshold Analysis**: Explore precision/recall trade-offs across different thresholds
- **Dataset Insights**: Label distribution, feature statistics, and token importance analysis
- **Robustness Testing**: Adversarial mutation generation and cross-model comparison

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
│LangGraph│ │ ML Models  │
│Pipeline │ │ (Joblib)   │
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
git clone https://github.com/Kartik200428/Minor_Project.git
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
│   │   └── main.py            
│   ├── core/
│   │   ├── config.py           
│   │   ├── model_loader.py      
│   │   ├── llm_manager.py      
│   │   ├── ti_manager.py        
│   │   ├── email_utils.py       
│   │   ├── eval_utils.py       
│   │   ├── dataset_utils.py     
│   │   └── robustness_utils.py   
│   ├── graph/
│   │   ├── graph_builder.py     
│   │   ├── nodes.py             
│   │   └── state.py             
│   └── models/
│       └── email_classifier.joblib 
├── frontend/
│   └── app.py                  
├── training/
│   ├── prepare_dataset.py      
│   └── train_classifier.py
|   └── spam_classifier.py       
├── data/
│   ├── raw/                     
│   ├── processed/               
│   └── ti/                      
├── test_graph.py                
├── requirements.txt             
└── README.md                   
```

## 🔌 API Endpoints

### Health & Configuration
- `GET /health` - Health check endpoint
- `GET /models` - List available LLM models

### Email Analysis
- `POST /analyze_email` - Analyze a single email through the full pipeline
- `POST /classify_spam` - Classify email as spam/ham using SVC model

### Evaluation
- `POST /eval_summary` - Run offline evaluation on processed dataset
- `POST /threshold_sweep` - Analyze performance across thresholds
- `POST /roc_curve` - Generate ROC curve data
- `POST /pr_curve` - Generate Precision-Recall curve data
- `POST /confusion_at_threshold` - Confusion matrix at specific threshold

### Dataset Insights
- `POST /dataset_summary` - High-level dataset statistics
- `POST /dataset_features` - Feature-level statistics and token analysis

### Robustness Testing
- `POST /adversarial_mutations` - Generate adversarial email variants
- `POST /cross_model_compare` - Compare outputs across different LLM models

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

## 🧪 Training

### Training the Phishing Classifier

1. **Prepare Dataset**:
   ```bash
   cd training
   python prepare_dataset.py
   ```
   This combines raw CSV files from `data/raw/` into `data/processed/combined.jsonl`

2. **Train Model**:
   ```bash
   python train_classifier.py
   ```
   This will:
   - Load the combined dataset
   - Train a TF-IDF + Logistic Regression classifier
   - Evaluate on a held-out test set
   - Save the model to `backend/models/email_classifier.joblib`

### Model Architecture

- **Vectorizer**: TF-IDF with max_features=60000, ngram_range=(1,2)
- **Classifier**: Logistic Regression with balanced class weights
- **Features**: Combined subject + body text
- **Labels**: `phishing` or `benign`

## ⚙️ Configuration

Configuration is centralized in `backend/core/config.py`:

- **LLM Models**: Available models and default selection
- **Heuristic Weights**: ML, heuristic, and TI score blending
- **Thresholds**: Phishing detection and high-risk thresholds
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
- **matplotlib**: Visualization for evaluation dashboard

## 📊 Evaluation Dashboard

The Streamlit UI includes a comprehensive evaluation dashboard with:

1. **Model Evaluation**: Classification metrics, confusion matrices
2. **Threshold Analysis**: Precision/recall trade-offs across thresholds
3. **Performance Curves**: ROC and Precision-Recall curves
4. **Dataset Insights**: Label distribution, feature statistics, token importance
5. **Robustness Testing**: Adversarial mutations and cross-model comparisons

## 🔒 Security Considerations

- **Local LLM Processing**: All LLM inference runs locally via Ollama (no external API calls)
- **Threat Intelligence**: URL reputation checks use public feeds (OpenPhish, URLHaus)
- **Model Artifacts**: Trained models are stored locally and not exposed via API
- **CORS**: Currently configured for local development (adjust for production)

## 🐛 Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check Ollama base URL in `backend/core/config.py`
- Verify models are pulled: `ollama list`

### Model Loading Errors
- Ensure `backend/models/email_classifier.joblib` exists
- Run training script if model is missing
- Check file paths in `backend/core/config.py`

### Dataset Not Found
- Run `training/prepare_dataset.py` to create `data/processed/combined.jsonl`
- Ensure raw datasets exist in `data/raw/`


##Author -- Kartik Singh




