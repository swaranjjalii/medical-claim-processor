<<<<<<< HEAD
# Medical Claim Processor

## Project overview

This project is a demo AI-powered pipeline to process medical insurance claim documents (medical bills, insurance ID cards, discharge summaries, prescriptions). It exposes a FastAPI endpoint (`/process-claim`) that accepts multiple PDF files and returns structured claim data plus an approval decision.

This repository contains a deterministic demo LLM client (mock) and production-ready hooks for real LLMs and OCR tools.

## Architecture & logic

High-level components:

- FastAPI app (`main.py`) — API entrypoint; `/process-claim` accepts uploaded files and returns a `ClaimProcessingResult`.
- ClaimProcessingOrchestrator — orchestrates pipeline steps:
  - Document extraction & classification
  - Per-document specialized processing (Bill, ID card, Discharge)
  - Cross-document consolidation
  - Validation
  - Decisioning
- Agents (single-responsibility components):
  - `DocumentClassifierAgent` — classifies documents into types (medical bill, id card, discharge, prescription). Uses lightweight heuristics first, falls back to LLM.
  - `TextExtractionAgent` — extracts text via `pdfplumber` and falls back to `pytesseract` OCR when needed. Also contains a safe fallback when those libraries are not installed.
  - `BillProcessingAgent` — extracts bill fields from bill text (hospital, amount, date, codes) using the LLM.
  - `IDCardProcessingAgent` — extracts patient fields from ID cards. Uses deterministic regex first and falls back to LLM if regex finds nothing.
  - `DischargeProcessingAgent` — extracts discharge-related fields using the LLM.
  - `ValidationAgent` — checks required fields and basic consistency.
  - `DecisionAgent` — auto-rejects on missing fields, otherwise uses LLM for nuanced decisions.
- LLM client abstraction (`GroqLLMClient`) — one place to wire a real LLM (Groq/OpenAI) or keep demo/mock behavior.

Processing flow (what happens when you POST files):
1. Extract text from each file using `TextExtractionAgent.extract` (pdfplumber -> OCR fallback -> raw bytes).
2. Classify each extracted doc with `DocumentClassifierAgent.classify` (heuristics -> LLM).
3. Run specialized agents per doc type to populate `BillData`, `PatientData`, `DischargeData`.
4. If patient fields are missing, run the ID parser on all documents (cross-document extraction) to catch patient info embedded in email/referral-like PDFs.
5. Validate structured data with `ValidationAgent`.
6. Make a decision with `DecisionAgent` (auto-reject if validation fails, or LLM decisioning).

## Where and how AI tools are used

- LLM usage points (in `main.py`):
  - `DocumentClassifierAgent.classify` — fallback classification using an LLM prompt when heuristics don't match.
  - `BillProcessingAgent.process` — ask LLM to extract billing fields and return JSON.
  - `IDCardProcessingAgent.process` — fallback to LLM if regex extraction fails.
  - `DischargeProcessingAgent.process` — extract discharge data using the LLM.
  - `DecisionAgent.decide` — for complex decision making and reasoning, the LLM is queried with the structured claim and validation results.
- LLM client: `GroqLLMClient.generate(prompt)` is the single place to call an LLM. In the demo, `GroqLLMClient` returns canned JSON for example prompts. Replace its `generate` implementation to call a real LLM API and parse results.

Notes on non-LLM AI:
- OCR: `pytesseract` (Tesseract OCR) for scanned documents. `pdfplumber` is used to extract embedded text first (faster and more accurate when available). `pdf2image` and image preprocessing can be added for improved OCR quality.

## Example prompts (actual examples used or suggested)

1) Document classifier (fallback prompt):

Prompt (sent to the LLM when heuristics don’t match):

{"Analyze this document and classify it into one of these types:\n- medical_bill\n- id_card\n- discharge_summary\n- prescription\n- unknown\n\nFilename: {filename}\nContent preview (first 500 chars): {text_preview[:500]}\n\nRespond with JSON: {"type": "document_type", "confidence": 0.0-1.0}"}

Example expected LLM response:

{"type": "id_card", "confidence": 0.98}

2) Bill extraction prompt (BillProcessingAgent):

Prompt:

"""Extract billing information from this medical bill text.\nReturn JSON with: hospital_name, total_amount, date_of_service, diagnosis_codes (list), procedure_codes (list)\n\nText: {text}\n\nJSON response:"""

Example expected response:

{"hospital_name": "City General Hospital", "total_amount": 15000.00, "date_of_service": "2024-10-15", "diagnosis_codes": ["J18.9"], "procedure_codes": ["99285"]}

3) ID card extraction fallback prompt (if regex misses):

Prompt:

"""Extract patient information from this insurance ID card.\nReturn JSON with: patient_name, patient_id, policy_number, date_of_birth\n\nText: {text}\n\nJSON response:"""

Example expected response:

{"patient_name": "John Doe", "patient_id": "PAT123456", "policy_number": "POL789012", "date_of_birth": "1985-03-15"}

4) Decision prompt (DecisionAgent uses LLM for nuanced cases):

Prompt:

"""Review this insurance claim and decide if it should be approved. Consider: - All required documents present - Valid diagnosis and procedure codes - Reasonable bill amount - No red flags\n\nClaim data: {structured_claim_json}\nValidation: {validation_json}\n\nRespond with JSON: {"decision": "approved/rejected/pending", "confidence": 0.0-1.0, "reasons": ["reason1", "reason2"]}"""

Example expected response:

{"decision": "approved", "confidence": 0.88, "reasons": ["All required documents present", "Bill amount within expected range"]}

## Vector store (bonus): how to integrate and why

Use case: If you want to improve extraction or decisioning using retrieval-augmented generation (RAG), add a vector store to:

- Store known templates (insurer-specific ID card templates, label variations).
- Store previously processed claims to surface similar cases for the DecisionAgent.
- Use an embedding model + FAISS/Chroma to retrieve relevant templates or past examples and include them in the LLM prompt.

Simple architecture:
- During ingestion, create an embedding for the extracted text or document (via OpenAI embeddings, or other embedding model).
- Store the vector with metadata (doc_type, source, insurer).
- During processing, retrieve top-K similar templates/examples and include them as context to the LLM prompt for extraction or decisioning.

Pseudo-code example (Python):

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

emb = OpenAIEmbeddings()
vs = FAISS.from_texts(["Fortis ID template text", "SBI ID template text"], emb)

# At runtime, get similar templates
query_emb = emb.embed_query(extracted_text)
results = vs.similarity_search_by_vector(query_emb, k=3)
# include results bits in the prompt for the LLM

Prompt snippet:
"Use these example templates to better identify labels in the following document:\n{retrieved_template_texts}\n\nNow extract patient_name and policy_number from:\n{text}\n"

Notes:
- For production use, consider a managed vector database (Pinecone, Milvus, Weaviate, Chroma Cloud) to scale.
- Keep PII off the public cloud unless properly encrypted and compliant with your policies.

## How to run locally

Prerequisites:
- Python 3.10+
- (Optional but recommended) Tesseract OCR installed on the machine for scanned PDFs.
  - Windows: install from UB Mannheim builds and add path to environment or set `TESSERACT_CMD`.

Install Python packages:

```powershell
pip install -r requirements.txt
```

Run the API:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Run the test client (uploads three dummy PDFs):

```powershell
python test_client.py
```

Extract locally for a single PDF (prints extracted text and parsed PatientData):

```powershell
python extract_local.py "C:\path\to\your.pdf"
```

If Tesseract is not on PATH, set it for the session before running Python:

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Files changed / important code locations

- `main.py` — core application and all agents. This is where the extraction, classification, parsing, validation, and decisioning logic lives.
- `extract_local.py` — small utility to run local extraction and ID parsing for debugging.
- `test_client.py` — test harness which POSTs files to `/process-claim` and prints results.

## Notes & next steps

- Replace the demo `GroqLLMClient.generate` implementation with a real LLM client for production.
- Optionally add `pdf2image` + preprocessing steps to improve OCR on low-quality scans.
- Add unit tests (pytest) around extraction and ID parsing for the most common insurers.

If you'd like, I can now:
- Add `pdf2image` rendering + preprocessing and wire it into the extractor, or
- Implement vector store integration with a quick FAISS/Chroma example and a small retrieval pipeline.

Pick one and I'll proceed.
=======
# medical-claim-processor
>>>>>>> 897ece77b671c863d4126bd38e0d429186e018bc
