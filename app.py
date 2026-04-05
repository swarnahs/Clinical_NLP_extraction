"""
app.py
------
Flask REST API exposing the clinical NLP extractor.

Endpoints:
    POST /api/extract          — extract from a single report text
    POST /api/extract_batch    — extract from multiple reports
    GET  /api/health           — health check

Run:
    python app.py
    # → http://localhost:5000
"""

import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from extractor import extract_report, extract_all_reports

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "engine": "rule-based NLP (no LLM)"})


@app.route("/api/extract", methods=["POST"])
def extract_single():
    """
    Body: { "text": "<report text>", "report_id": "Report 1" }
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    report_id = data.get("report_id", "Report")
    result    = extract_report(text, report_id)
    return jsonify(result)


@app.route("/api/extract_batch", methods=["POST"])
def extract_batch():
    """
    Body: { "reports": [{"id": "Report 1", "text": "..."}, ...] }
    """
    data    = request.get_json(force=True)
    reports = data.get("reports", [])
    if not reports:
        return jsonify({"error": "No reports provided"}), 400

    results = extract_all_reports(reports)
    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)