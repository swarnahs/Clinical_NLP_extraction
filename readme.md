# Clinical NLP Extractor 🏥

Extract structured clinical information from endoscopy reports using **traditional NLP only** — no LLMs, no OpenAI, no GPT.

## Techniques Used
- Regex pattern matching (ICD-10, CPT, HCPCS codes)
- Curated medical lexicons for NER (Named Entity Recognition)
- Section-boundary detection via keyword anchors
- Multi-word phrase matching with n-gram windows
- Rule-based sentence tokenisation

## Project Structure

```
med_ass/
├── app.py              ← Flask REST API
├── extractor.py        ← Core Python NLP engine
├── index.html          ← Standalone web UI
├── requirements.txt    ← Python dependencies
├── Procfile            ← For deployment (Render/Railway/Heroku)
└── .gitignore
```

## Local Usage

### Option 1 — Browser (no install)
Just open `index.html` in any browser. Load a sample report or paste your own.

### Option 2 — Python CLI
```bash
pip install -r requirements.txt
python extractor.py    # runs a sample extraction
```

### Option 3 — Flask API (local)
```bash
pip install -r requirements.txt
python app.py          # → http://localhost:5000
```

#### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/extract` | Extract from a single report |
| POST | `/api/extract_batch` | Extract from multiple reports |

```bash
# Extract a single report
curl -X POST http://localhost:5000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Diagnosis: K64.8 - Internal hemorrhoids", "report_id": "Report 1"}'
```

## Output Format

```json
{
  "ReportID": "Report 1",
  "Clinical Terms": ["Internal hemorrhoids", "Melanosis coli"],
  "Anatomical Locations": ["Rectum", "Sigmoid colon", "Cecum"],
  "Diagnosis": ["Internal hemorrhoids", "Diverticulosis (sigmoid)"],
  "Procedures": ["Colonoscopy"],
  "ICD-10": ["K57.90", "K64.8", "Z86.0100"],
  "CPT": ["45378"],
  "HCPCS": [],
  "Modifiers": []
}
```

## Deployment

### Deploy to Render (free)
1. Push code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set **Start Command**: `gunicorn app:app`
5. Set **Environment**: Python 3

### Deploy to Railway
1. Push code to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo — it auto-detects the `Procfile`