"""Utility to extract text from a local PDF using the project's TextExtractionAgent.

Usage:
    python extract_local.py "C:\path\to\file.pdf"

This reuses the same extraction logic (pdfplumber + pytesseract fallback) implemented in
`main.TextExtractionAgent` so you can test exactly how uploaded PDFs will be parsed.
"""
import sys
import asyncio
from pathlib import Path

from main import GroqLLMClient, TextExtractionAgent, IDCardProcessingAgent, SimpleVectorStore, build_default_templates

async def extract_file(path: Path):
    if not path.exists():
        print(f"File not found: {path}")
        return
    with path.open('rb') as f:
        raw = f.read()

    llm = GroqLLMClient(api_key="dummy")
    extractor = TextExtractionAgent(llm)
    text = await extractor.extract(raw, path.name)
    print("\n--- Extracted Text Start ---\n")
    print(text)
    print("\n--- Extracted Text End ---\n")
    # Also run the ID card parser to show what fields would be extracted
    # Construct a small vector store like the server does and pass it to the ID agent
    vs = SimpleVectorStore()
    templates = build_default_templates()
    vs.add_texts([t["text"] for t in templates], [t.get("metadata", {}) for t in templates])
    id_agent = IDCardProcessingAgent(llm, vector_store=vs)
    pd = await id_agent.process(text)
    print("\n--- Parsed PatientData ---\n")
    print(pd.model_dump())
    print("\n--- End Parsed PatientData ---\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_local.py <path-to-pdf>")
        sys.exit(1)
    p = Path(sys.argv[1])
    asyncio.run(extract_file(p))
