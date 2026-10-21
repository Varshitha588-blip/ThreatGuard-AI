# ThreatGuard AI
### AI-Powered Phishing Message and Synthetic Voice Detection

ThreatGuard AI is a full-stack security prototype that analyzes suspicious text messages and audio recordings using machine-learning models. It combines model predictions with transparent, rule-based indicators to help users investigate potentially malicious content.

> **Research prototype:** Predictions are estimates, not proof that a message is malicious or that a recording is authentic.

## Features

- **Phishing detection:** Fine-tuned DistilBERT text classification.
- **Voice-spoof detection:** Wav2Vec2 embeddings followed by a scikit-learn classifier.
- **Text indicators:** Identifies patterns involving urgency, credentials, financial pressure, authority language, and URLs.
- **Audio explanations:** Reports the classifier output without claiming to identify specific acoustic defects.
- **REST API:** FastAPI endpoints for health checks, text analysis, and audio uploads.
- **Web interface:** React and Vite frontend.
- **Automated tests:** API and detector tests using pytest.

## Architecture

```text
React + Vite Frontend
        |
        | HTTP requests
        v
FastAPI Backend
        |
        +--> Text Analysis
        |       |
        |       +--> Fine-tuned DistilBERT
        |       +--> Rule-based text indicators
        |
        +--> Audio Analysis
                |
                +--> Audio decoding and resampling
                +--> Wav2Vec2 embeddings (768 dimensions)
                +--> StandardScaler + Logistic Regression
```

## Technology Stack

| Area | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI, Pydantic |
| Text model | PyTorch, Hugging Face Transformers, DistilBERT |
| Audio model | Wav2Vec2, librosa, NumPy, scikit-learn, joblib |
| Testing | pytest, FastAPI TestClient |
| Model source | Hugging Face model repository or local model files |

## Project Structure

```text
ai-phishing-deepfake-detector/
├── app/
│   ├── main.py
│   ├── services/
│   │   ├── text_detector.py
│   │   ├── audio_detector.py
│   │   └── explanation.py
│   ├── tasks/
│   └── workflows/
├── frontend/
│   └── src/
├── models/                 # Local models; excluded from Git
├── data/                   # Datasets and features; excluded from Git
├── reports/                # Evaluation reports
├── tests/
├── training/
├── .env.example
├── .gitignore
└── requirements.txt
```

## Requirements

- Python 3.12 recommended for the current development environment.
- Node.js and npm compatible with the Vite version in `frontend/package.json`.
- Internet access for downloading pretrained models when they are not cached locally.
- Sufficient memory to load the Transformer models.

## Local Setup — Windows PowerShell

Run these commands from the repository root.

### 1. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks activation, use this alternative for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure the model paths

The default text model directory is:

```text
models/text/distilbert-phishing/
```

It must contain the trained model configuration, tokenizer files, and model weights.

The audio classifier defaults to:

```text
models/audio/cross_generator_wav2vec2_classifier.joblib
```

The Wav2Vec2 encoder defaults to the public pretrained model `facebook/wav2vec2-base`.

**Important:** model weights and datasets are excluded from this Git repository. A fresh clone will not contain the trained text model or custom audio classifier unless you download or copy them separately.

### 4. Start the backend

From the repository root, with the virtual environment active:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

Check the health endpoint at:

```text
http://127.0.0.1:8000/health
```

### 5. Start the frontend

Open a second PowerShell terminal:

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev
```

Open the local URL printed by Vite, normally:

```text
http://localhost:5173
```

## Environment Configuration

The backend reads these optional environment variables:

| Variable | Purpose |
|---|---|
| `CORS_ORIGINS` | Comma-separated list of permitted frontend origins |
| `HF_TOKEN` | Optional token for accessing private Hugging Face repositories |
| `TEXT_MODEL_ID` | Hugging Face repository ID for the trained text model |
| `AUDIO_MODEL_ID` | Hugging Face repository ID for the Wav2Vec2 encoder; defaults to `facebook/wav2vec2-base` |
| `VITE_API_BASE_URL` | Frontend build-time URL of the backend API |

The `.env.example` file documents configuration names. It does not automatically load environment variables. Configure them in your terminal or hosting platform, or explicitly add environment-file loading if desired.

Never commit real tokens, passwords, private keys, or `.env` files.

**Deployment note:** `TEXT_MODEL_ID` can select a remote text model, but the audio classifier currently loads from a local path. A hosted deployment must provide that classifier file through secure model storage or a suitable mounted volume. The model-loading strategy must be configured before deployment.

## API Reference

### `GET /`

Returns a short API status message and the documentation path.

### `GET /health`

Returns:

```json
{
  "status": "ok"
}
```

This checks API availability; it does not verify that both trained models have loaded successfully.

### `POST /analyze/text`

Example request:

```json
{
  "text": "Urgent: verify your account immediately."
}
```

The response contains the predicted label, phishing and legitimate probabilities, rule-based explanation, and a warning about model limitations.

### `POST /analyze/audio`

Upload a multipart form file using the field name `file`. Supported filename extensions are `.wav`, `.mp3`, `.flac`, and `.ogg`. The API limits uploads to 20 MB.

The response includes the predicted `genuine` or `spoof` label, classifier probabilities when available, audio metadata, an explanation, and a warning.

The current endpoint checks the filename extension and size; these checks alone do not establish that a file is valid audio.

## Evaluation Results

Results below are from the saved local evaluation reports.

### Text classifier

- Model: fine-tuned DistilBERT.
- Test samples: 1,744.
- Accuracy: **99.20%**.
- Phishing precision: 99.08%.
- Phishing recall: 98.77%.
- Phishing F1 score: 98.92%.

Confusion matrix, with rows representing actual classes and columns representing predicted classes in the order legitimate, phishing:

```text
[[1086, 6],
 [   8, 644]]
```

### Audio classifier

- Encoder: `facebook/wav2vec2-base`.
- Features: mean-pooled embeddings of 768 dimensions.
- Classifier: `StandardScaler` + `LogisticRegression`.

| Test split | Recordings | Accuracy |
|---|---:|---:|
| Seen-generator test | 1,436 | 99.44% |
| Unseen-generator test | 1,436 | 99.65% |

The unseen-generator split evaluates one held-out spoof-generation setup in the documented dataset. It does not demonstrate generalization to every voice, language, recording condition, or synthesis system.

### Reproducibility

The saved metrics are available locally under `reports/`, including:

- `reports/text_model_evaluation.json`
- `reports/transformer/test_metrics.json`
- `reports/audio/cross_generator_audio_evaluation.json`

Datasets, checkpoints, prediction exports, and trained weights are excluded from Git. Reproducing the results requires the appropriate dataset, preprocessing, and model artifacts.

## Run Tests

From the repository root with the Python virtual environment activated:

```powershell
python -m pytest tests -v
```

The current test suite covers API responses, text prediction behavior with mocked model outputs, missing-model error handling, and audio upload validation and response handling.

Some tests mock model predictions. Passing those tests does not by itself verify the quality of the trained models or the behavior of every real audio file.

## Limitations and Responsible Use

- Predictions are model estimates and may be incorrect.
- High benchmark accuracy does not guarantee real-world performance.
- Text indicators are rule-based observations, not causal explanations of the Transformer prediction.
- URLs are not automatically opened or verified by the text explanation rules.
- Audio embeddings do not directly identify human-readable voice characteristics.
- The audio evaluation covers specific dataset splits and a limited generator setup.
- Do not use this prototype as the sole basis for financial, legal, employment, or other high-impact decisions.
- Avoid uploading private, confidential, or personally identifying messages and recordings to an untrusted deployment.

## Future Improvements

- Secure, reproducible model artifact downloads for deployment.
- Integration tests using actual model artifacts and representative audio files.
- Audio-content validation and stricter upload handling.
- Broader evaluation across languages, devices, datasets, and unseen generation systems.
- Monitoring, model versioning, and documented reproducible training pipelines.

## License and Dataset Attribution

Choose and add an appropriate project license before distributing this repository. Verify the licenses and usage terms of each dataset, pretrained encoder, and model artifact before redistributing them. Dataset attribution and license details should be recorded here before public release.

---

**ThreatGuard AI** — an educational security project for investigating phishing messages and synthetic voice recordings.

