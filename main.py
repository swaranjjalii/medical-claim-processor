"""
Medical Insurance Claim Document Processor
A production-ready agentic backend pipeline using FastAPI, LangGraph, and Groq
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import io
import re
import os
try:
    import pdfplumber
    from PIL import Image
    import pytesseract
except Exception:
    # Optional dependencies may not be installed in the environment used for tests.
    pdfplumber = None
    pytesseract = None
    Image = None
else:
    # If pytesseract is available, try to configure the tesseract executable path from
    # environment or common installation locations on Windows.
    try:
        if pytesseract is not None:
            tcmd = os.environ.get("TESSERACT_CMD")
            if not tcmd:
                # Common Windows install locations
                candidates = [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
                for c in candidates:
                    if os.path.exists(c):
                        tcmd = c
                        break
            if tcmd:
                try:
                    pytesseract.pytesseract.tesseract_cmd = tcmd
                    logger = logging.getLogger(__name__)
                    logger.info(f"Configured pytesseract.tesseract_cmd = {tcmd}")
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Could not set pytesseract cmd: {e}")
    except Exception:
        # Non-fatal: continue without explicit tesseract_cmd configuration
        pass
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import logging
from datetime import datetime
import json
from vector_store import SimpleVectorStore, build_default_templates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODELS & SCHEMAS
# ============================================================================

class DocumentType(str, Enum):
    MEDICAL_BILL = "medical_bill"
    ID_CARD = "id_card"
    DISCHARGE_SUMMARY = "discharge_summary"
    PRESCRIPTION = "prescription"
    UNKNOWN = "unknown"

class ClaimDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"

class ExtractedDocument(BaseModel):
    filename: str
    doc_type: DocumentType
    extracted_text: str
    confidence: float = Field(ge=0, le=1)

class BillData(BaseModel):
    hospital_name: Optional[str] = None
    total_amount: Optional[float] = None
    date_of_service: Optional[str] = None
    diagnosis_codes: List[str] = []
    procedure_codes: List[str] = []

class PatientData(BaseModel):
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    policy_number: Optional[str] = None
    date_of_birth: Optional[str] = None

class DischargeData(BaseModel):
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    diagnosis: Optional[str] = None
    treatment_summary: Optional[str] = None
    attending_physician: Optional[str] = None

class StructuredClaim(BaseModel):
    bill_data: BillData
    patient_data: PatientData
    discharge_data: DischargeData
    timestamp: str

class ValidationResult(BaseModel):
    is_valid: bool
    missing_fields: List[str] = []
    inconsistencies: List[str] = []
    warnings: List[str] = []

class ClaimProcessingResult(BaseModel):
    claim_id: str
    decision: ClaimDecision
    confidence: float
    structured_data: StructuredClaim
    validation: ValidationResult
    reasons: List[str]
    processing_time_seconds: float

# ============================================================================
# AGENTS
# ============================================================================

class DocumentClassifierAgent:
    """Classifies documents based on content and filename"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def classify(self, filename: str, text_preview: str) -> tuple[DocumentType, float]:
        """Classify document type with confidence score"""
        # Lightweight heuristic classification for demo/testing.
        # Prefer deterministic rules over LLM calls so demo files are classified reliably.
        fname = (filename or "").lower()
        preview = (text_preview or "").lower()

        # Filename-based heuristics
        if any(k in fname for k in ("id_card", "idcard", "id-", "insurance_id", "insurance")) or "id card" in preview:
            return DocumentType.ID_CARD, 0.98
        if "discharge" in fname or "discharge" in preview:
            return DocumentType.DISCHARGE_SUMMARY, 0.97
        if "prescription" in fname or "rx" in preview or "prescription" in preview:
            return DocumentType.PRESCRIPTION, 0.95
        if any(k in fname for k in ("bill", "invoice")) or any(k in preview for k in ("total amount", "amount due", "invoice", "bill")):
            return DocumentType.MEDICAL_BILL, 0.96

        # Fallback to LLM classification when heuristics don't match
        prompt = f"""Analyze this document and classify it into one of these types:
- medical_bill: Hospital bills, invoices with charges
- id_card: Insurance ID cards, membership cards
- discharge_summary: Hospital discharge papers, summary of treatment
- prescription: Medicine prescriptions
- unknown: If unclear

Filename: {filename}
Content preview (first 500 chars): {text_preview[:500]}

Respond with JSON: {{"type": "document_type", "confidence": 0.0-1.0}}"""

        try:
            response = await self.llm.generate(prompt)
            result = json.loads(response)
            doc_type = DocumentType(result.get("type", "unknown"))
            confidence = float(result.get("confidence", 0.5))
            return doc_type, confidence
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return DocumentType.UNKNOWN, 0.3

class TextExtractionAgent:
    """Extracts and cleans text from documents"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def extract(self, raw_content: bytes, filename: str) -> str:
        """Extract text from document bytes (simulated for demo)"""
        # Prefer actual PDF text extraction when dependencies are available.
        # pdfplumber will extract embedded text; if that fails or returns very little
        # text (scanned PDF), fall back to OCR via pytesseract.
        try:
            if pdfplumber is None:
                # Dependencies not installed; return a simple simulated extract
                logger.debug("pdfplumber/pytesseract not available, using simulated extraction")
                text = f"[Extracted from {filename}]\n"
                text += "Sample medical document content would appear here after OCR/PDF parsing."
                return text

            buf = io.BytesIO(raw_content)
            full_text = ""
            with pdfplumber.open(buf) as pdf:
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    full_text += page_text + "\n"

                # If extracted text is very short, attempt OCR on each page
                if len(full_text.strip()) < 50 and pytesseract is not None:
                    logger.info("Low text extracted from PDF, running OCR fallback")
                    ocr_text = ""
                    for page in pdf.pages:
                        try:
                            pil_img = page.to_image(resolution=300).original
                            page_ocr = pytesseract.image_to_string(pil_img)
                            ocr_text += page_ocr + "\n"
                        except Exception as e:
                            logger.warning(f"OCR failed for a page: {e}")
                    if ocr_text.strip():
                        full_text = ocr_text

            # If still empty, try a raw decode as a last resort
            if not full_text.strip():
                try:
                    full_text = raw_content.decode(errors="ignore")
                except Exception:
                    full_text = f"[Unable to extract textual content from {filename}]"

            return full_text
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

class BillProcessingAgent:
    """Processes medical bill documents"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def process(self, text: str) -> BillData:
        """Extract structured data from bill text"""
        prompt = f"""Extract billing information from this medical bill text.
Return JSON with: hospital_name, total_amount, date_of_service, diagnosis_codes (list), procedure_codes (list)

Text: {text}

JSON response:"""

        try:
            response = await self.llm.generate(prompt)
            data = json.loads(response)
            return BillData(**data)
        except Exception as e:
            logger.warning(f"Bill processing error: {e}")
            return BillData()

class IDCardProcessingAgent:
    """Processes insurance ID card documents"""
    
    def __init__(self, llm_client, vector_store: Optional[SimpleVectorStore] = None):
        self.llm = llm_client
        self.vector_store = vector_store
    
    async def process(self, text: str) -> PatientData:
        """Extract patient information from ID card"""
        # Try fast regex-based extraction first (deterministic and works well for many ID cards)
        # Patterns target common label forms found on insurance ID cards
        try:
            pd = PatientData()
            # Normalize line endings and split
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            joined = "\n".join(lines)

            # Patient name heuristics - allow forms like "Patient name Mary Philo (Female)" or "Name: John Doe"
            name_match = re.search(r"(?:patient\s*name|name|insured|member)\s*(?:[:\-])?\s*(.+)", joined, re.IGNORECASE)
            if name_match:
                name_val = name_match.group(1).strip()
                # strip trailing parenthetical like (Female)
                name_val = re.sub(r"\s*\(.*\)$", "", name_val).strip()
                pd.patient_name = name_val

            # Policy number heuristics
            # Policy number heuristics - allow forms with or without colon, e.g. "Policy Number 4101..."
            # Require the captured policy value to contain at least one digit to avoid matching nearby words like 'terms'
            policy_match = re.search(r"(?:policy(?:\snumber|\sno(?:\.|)?|\s#)?|policy number)\s*[:\-]?\s*([A-Z0-9\-./]*\d[A-Z0-9\-./]*)", joined, re.IGNORECASE)
            if policy_match:
                pd.policy_number = policy_match.group(1).strip()

            # Patient ID heuristics
            id_match = re.search(r"(?:member\s*id|patient\s*id|id\s*no(?:\.|)?)\s*[:\-]?\s*([A-Z0-9\-./]+)", joined, re.IGNORECASE)
            if id_match:
                pd.patient_id = id_match.group(1).strip()

            # Date of birth heuristics
            dob_match = re.search(r"(?:DOB|Date of Birth)\s*[:\-]\s*([0-9]{1,2}[\-/ ][A-Za-z0-9]{1,11}[\-/ ][0-9]{2,4})", joined, re.IGNORECASE)
            if dob_match:
                pd.date_of_birth = dob_match.group(1).strip()

            # If any of the key fields were found, return them
            if pd.patient_name or pd.policy_number or pd.patient_id or pd.date_of_birth:
                return pd

            # Fallback to LLM if regex did not find anything
            prompt = f"""Extract patient information from this insurance ID card.
Return JSON with: patient_name, patient_id, policy_number, date_of_birth

Text: {text}

JSON response:"""
            response = await self.llm.generate(prompt)
            data = json.loads(response)
            return PatientData(**data)
        except Exception as e:
            logger.warning(f"ID card processing error: {e}")
            return PatientData()

class DischargeProcessingAgent:
    """Processes discharge summary documents"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def process(self, text: str) -> DischargeData:
        """Extract discharge information"""
        prompt = f"""Extract discharge information from this hospital discharge summary.
Return JSON with: admission_date, discharge_date, diagnosis, treatment_summary, attending_physician

Text: {text}

JSON response:"""

        try:
            response = await self.llm.generate(prompt)
            data = json.loads(response)
            return DischargeData(**data)
        except Exception as e:
            logger.warning(f"Discharge processing error: {e}")
            return DischargeData()

class ValidationAgent:
    """Validates structured claim data for completeness and consistency"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def validate(self, structured_claim: StructuredClaim) -> ValidationResult:
        """Validate claim data"""
        missing = []
        inconsistencies = []
        warnings = []
        
        # Check required fields
        if not structured_claim.patient_data.patient_name:
            missing.append("patient_name")
        if not structured_claim.patient_data.policy_number:
            missing.append("policy_number")
        if not structured_claim.bill_data.total_amount:
            missing.append("total_amount")
        if not structured_claim.bill_data.hospital_name:
            missing.append("hospital_name")
        
        # Check for inconsistencies
        if structured_claim.discharge_data.admission_date and \
           structured_claim.bill_data.date_of_service:
            # In real implementation, validate dates
            pass
        
        if structured_claim.bill_data.total_amount and \
           structured_claim.bill_data.total_amount > 100000:
            warnings.append("High claim amount - requires manual review")
        
        is_valid = len(missing) == 0 and len(inconsistencies) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            missing_fields=missing,
            inconsistencies=inconsistencies,
            warnings=warnings
        )

class DecisionAgent:
    """Makes final claim approval/rejection decision"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def decide(
        self, 
        structured_claim: StructuredClaim, 
        validation: ValidationResult
    ) -> tuple[ClaimDecision, float, List[str]]:
        """Make claim decision with reasoning"""
        
        reasons = []
        
        # Auto-reject if validation failed
        if not validation.is_valid:
            reasons.append("Missing required fields: " + ", ".join(validation.missing_fields))
            if validation.inconsistencies:
                reasons.append("Data inconsistencies found: " + ", ".join(validation.inconsistencies))
            return ClaimDecision.REJECTED, 0.95, reasons
        
        # Check warnings
        if validation.warnings:
            reasons.extend(validation.warnings)
            return ClaimDecision.PENDING, 0.70, reasons
        
        # Use LLM for complex decision making
        prompt = f"""Review this insurance claim and decide if it should be approved.
Consider:
- All required documents present
- Valid diagnosis and procedure codes
- Reasonable bill amount
- No red flags

Claim data: {structured_claim.model_dump_json()}
Validation: {validation.model_dump_json()}

Respond with JSON: {{"decision": "approved/rejected/pending", "confidence": 0.0-1.0, "reasons": ["reason1", "reason2"]}}"""

        try:
            response = await self.llm.generate(prompt)
            result = json.loads(response)
            decision = ClaimDecision(result.get("decision", "pending"))
            confidence = float(result.get("confidence", 0.5))
            reasons = result.get("reasons", ["Automated review completed"])
            return decision, confidence, reasons
        except Exception as e:
            logger.error(f"Decision error: {e}")
            return ClaimDecision.PENDING, 0.5, ["Manual review required due to processing error"]

# ============================================================================
# LLM CLIENT (Groq Integration)
# ============================================================================

class GroqLLMClient:
    """Groq LLM client for fast inference"""
    
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        self.api_key = api_key
        self.model = model
        # In production, initialize actual Groq client
        # from groq import AsyncGroq
        # self.client = AsyncGroq(api_key=api_key)
    
    async def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate response from LLM"""
        # Simulated response for demo
        # In production, call actual Groq API
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Return mock structured responses based on prompt
        if "classify" in prompt.lower():
            return json.dumps({"type": "medical_bill", "confidence": 0.92})
        elif "billing information" in prompt.lower():
            return json.dumps({
                "hospital_name": "City General Hospital",
                "total_amount": 15000.00,
                "date_of_service": "2024-10-15",
                "diagnosis_codes": ["J18.9", "R50.9"],
                "procedure_codes": ["99285", "71020"]
            })
        elif "patient information" in prompt.lower():
            return json.dumps({
                "patient_name": "John Doe",
                "patient_id": "PAT123456",
                "policy_number": "POL789012",
                "date_of_birth": "1985-03-15"
            })
        elif "discharge information" in prompt.lower():
            return json.dumps({
                "admission_date": "2024-10-12",
                "discharge_date": "2024-10-18",
                "diagnosis": "Pneumonia with complications",
                "treatment_summary": "Patient treated with IV antibiotics and respiratory therapy",
                "attending_physician": "Dr. Sarah Smith"
            })
        elif "Review this insurance claim" in prompt:
            return json.dumps({
                "decision": "approved",
                "confidence": 0.88,
                "reasons": [
                    "All required documents present",
                    "Valid diagnosis and procedure codes match",
                    "Bill amount within expected range for treatment",
                    "No inconsistencies detected"
                ]
            })
        else:
            return json.dumps({"status": "processed"})

# ============================================================================
# ORCHESTRATOR (LangGraph-style workflow)
# ============================================================================

class ClaimProcessingOrchestrator:
    """Orchestrates the multi-agent claim processing pipeline"""
    
    def __init__(self, llm_client):
        # Build a small retrieval store for templates/examples and pass it into agents.
        self.vector_store = SimpleVectorStore()
        templates = build_default_templates()
        self.vector_store.add_texts([t["text"] for t in templates], [t.get("metadata", {}) for t in templates])

        self.classifier = DocumentClassifierAgent(llm_client)
        self.extractor = TextExtractionAgent(llm_client)
        self.bill_agent = BillProcessingAgent(llm_client)
        self.id_agent = IDCardProcessingAgent(llm_client, vector_store=self.vector_store)
        self.discharge_agent = DischargeProcessingAgent(llm_client)
        self.validator = ValidationAgent(llm_client)
        self.decision_agent = DecisionAgent(llm_client)
    
    async def process_documents(
        self, 
        files: List[UploadFile]
    ) -> ClaimProcessingResult:
        """Main processing pipeline"""
        start_time = datetime.now()
        
        # Step 1: Extract and classify all documents
        logger.info(f"Processing {len(files)} documents")
        extracted_docs = await self._extract_and_classify(files)
        
        # Step 2: Process documents by type using specialized agents
        logger.info("Processing documents with specialized agents")
        bill_data = BillData()
        patient_data = PatientData()
        discharge_data = DischargeData()
        
        # Process each document with appropriate agent
        tasks = []
        for doc in extracted_docs:
            if doc.doc_type == DocumentType.MEDICAL_BILL:
                tasks.append(self._process_bill(doc, bill_data))
            elif doc.doc_type == DocumentType.ID_CARD:
                tasks.append(self._process_id_card(doc, patient_data))
            elif doc.doc_type == DocumentType.DISCHARGE_SUMMARY:
                tasks.append(self._process_discharge(doc, discharge_data))
        
        await asyncio.gather(*tasks)

        # If patient info wasn't found in ID card docs, attempt to extract patient info
        # from any document text (some PDFs are emails/referrals that contain patient/policy lines)
        if not patient_data.patient_name or not patient_data.policy_number:
            # Create tasks to run ID card parser on all documents where we didn't already parse patient info
            id_tasks = []
            for doc in extracted_docs:
                id_tasks.append(self.id_agent.process(doc.extracted_text))
            id_results = await asyncio.gather(*id_tasks)
            for res in id_results:
                # Handle either PatientData model or dict-like response
                try:
                    # pydantic model
                    pname = getattr(res, "patient_name", None)
                    ppolicy = getattr(res, "policy_number", None)
                    pid = getattr(res, "patient_id", None)
                    pdob = getattr(res, "date_of_birth", None)
                except Exception:
                    # dict-like
                    pname = res.get("patient_name") if isinstance(res, dict) else None
                    ppolicy = res.get("policy_number") if isinstance(res, dict) else None
                    pid = res.get("patient_id") if isinstance(res, dict) else None
                    pdob = res.get("date_of_birth") if isinstance(res, dict) else None

                if not patient_data.patient_name and pname:
                    patient_data.patient_name = pname
                if not patient_data.policy_number and ppolicy:
                    patient_data.policy_number = ppolicy
                if not patient_data.patient_id and pid:
                    patient_data.patient_id = pid
                if not patient_data.date_of_birth and pdob:
                    patient_data.date_of_birth = pdob
        
        # Step 3: Structure the claim
        structured_claim = StructuredClaim(
            bill_data=bill_data,
            patient_data=patient_data,
            discharge_data=discharge_data,
            timestamp=datetime.now().isoformat()
        )
        
        # Step 4: Validate the structured data
        logger.info("Validating claim data")
        validation = await self.validator.validate(structured_claim)
        
        # Step 5: Make final decision
        logger.info("Making claim decision")
        decision, confidence, reasons = await self.decision_agent.decide(
            structured_claim, validation
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate unique claim ID
        claim_id = f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return ClaimProcessingResult(
            claim_id=claim_id,
            decision=decision,
            confidence=confidence,
            structured_data=structured_claim,
            validation=validation,
            reasons=reasons,
            processing_time_seconds=round(processing_time, 2)
        )
    
    async def _extract_and_classify(
        self, 
        files: List[UploadFile]
    ) -> List[ExtractedDocument]:
        """Extract text and classify documents concurrently"""
        async def process_file(file: UploadFile) -> ExtractedDocument:
            content = await file.read()
            text = await self.extractor.extract(content, file.filename)
            doc_type, confidence = await self.classifier.classify(file.filename, text)
            return ExtractedDocument(
                filename=file.filename,
                doc_type=doc_type,
                extracted_text=text,
                confidence=confidence
            )
        
        tasks = [process_file(f) for f in files]
        return await asyncio.gather(*tasks)
    
    async def _process_bill(self, doc: ExtractedDocument, bill_data: BillData):
        """Process bill document"""
        result = await self.bill_agent.process(doc.extracted_text)
        # Update bill_data with results
        bill_data.hospital_name = result.hospital_name
        bill_data.total_amount = result.total_amount
        bill_data.date_of_service = result.date_of_service
        bill_data.diagnosis_codes = result.diagnosis_codes
        bill_data.procedure_codes = result.procedure_codes
    
    async def _process_id_card(self, doc: ExtractedDocument, patient_data: PatientData):
        """Process ID card document"""
        result = await self.id_agent.process(doc.extracted_text)
        patient_data.patient_name = result.patient_name
        patient_data.patient_id = result.patient_id
        patient_data.policy_number = result.policy_number
        patient_data.date_of_birth = result.date_of_birth
    
    async def _process_discharge(self, doc: ExtractedDocument, discharge_data: DischargeData):
        """Process discharge summary document"""
        result = await self.discharge_agent.process(doc.extracted_text)
        discharge_data.admission_date = result.admission_date
        discharge_data.discharge_date = result.discharge_date
        discharge_data.diagnosis = result.diagnosis
        discharge_data.treatment_summary = result.treatment_summary
        discharge_data.attending_physician = result.attending_physician

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Medical Insurance Claim Processor",
    description="AI-powered agentic pipeline for processing medical insurance claims",
    version="1.0.0"
)

# Initialize LLM client and orchestrator
# In production, load API key from environment
llm_client = GroqLLMClient(api_key="your-groq-api-key-here")
orchestrator = ClaimProcessingOrchestrator(llm_client)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Medical Claim Processor",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post("/process-claim", response_model=ClaimProcessingResult)
async def process_claim(
    files: List[UploadFile] = File(..., description="Upload medical documents (PDFs)")
):
    """
    Process medical insurance claim documents
    
    Accepts multiple PDF files including:
    - Medical bills/invoices
    - Insurance ID cards
    - Discharge summaries
    - Prescriptions
    
    Returns structured claim data with approval decision
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    # Validate file types
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")
        # In production, validate MIME type
        # if not file.content_type == "application/pdf":
        #     raise HTTPException(status_code=400, detail=f"File {file.filename} must be PDF")
    
    try:
        logger.info(f"Starting claim processing for {len(files)} files")
        result = await orchestrator.process_documents(files)
        logger.info(f"Claim {result.claim_id} processed: {result.decision}")
        return result
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)