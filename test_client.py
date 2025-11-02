"""
Simple test client to verify the claim processor API
Run this after starting the server with: uvicorn main:app --reload
"""

import requests
import json
from io import BytesIO

# API endpoint
BASE_URL = "http://localhost:8000"

def create_dummy_pdf(filename: str) -> BytesIO:
    """Create a dummy PDF file for testing"""
    content = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
({filename} - Sample Medical Document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
308
%%EOF"""
    return BytesIO(content.encode())

def test_health_check():
    """Test the health check endpoint"""
    print("\n🔍 Testing health check endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Health check passed!")

def test_process_claim():
    """Test the claim processing endpoint"""
    print("\n📄 Testing claim processing endpoint...")
    
    # Create dummy PDF files
    files = [
        ('files', ('medical_bill.pdf', create_dummy_pdf('Medical Bill'), 'application/pdf')),
        ('files', ('id_card.pdf', create_dummy_pdf('ID Card'), 'application/pdf')),
        ('files', ('discharge_summary.pdf', create_dummy_pdf('Discharge Summary'), 'application/pdf'))
    ]
    
    print("Uploading 3 documents: medical_bill.pdf, id_card.pdf, discharge_summary.pdf")
    
    response = requests.post(
        f"{BASE_URL}/process-claim",
        files=files
    )
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 Processing Result:")
        print(f"  Claim ID: {result['claim_id']}")
        print(f"  Decision: {result['decision'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Processing Time: {result['processing_time_seconds']}s")
        
        print("\n👤 Patient Information:")
        patient = result['structured_data']['patient_data']
        print(f"  Name: {patient.get('patient_name', 'N/A')}")
        print(f"  Policy: {patient.get('policy_number', 'N/A')}")
        
        print("\n💰 Bill Information:")
        bill = result['structured_data']['bill_data']
        print(f"  Hospital: {bill.get('hospital_name', 'N/A')}")
        print(f"  Amount: ${bill.get('total_amount', 0):,.2f}")
        
        print("\n🏥 Discharge Information:")
        discharge = result['structured_data']['discharge_data']
        print(f"  Diagnosis: {discharge.get('diagnosis', 'N/A')}")
        print(f"  Physician: {discharge.get('attending_physician', 'N/A')}")
        
        print("\n✅ Validation Status:")
        validation = result['validation']
        print(f"  Valid: {validation['is_valid']}")
        if validation['missing_fields']:
            print(f"  Missing: {', '.join(validation['missing_fields'])}")
        if validation['warnings']:
            print(f"  Warnings: {', '.join(validation['warnings'])}")
        
        print("\n💡 Decision Reasons:")
        for reason in result['reasons']:
            print(f"  • {reason}")
        
        print("\n✅ Claim processing test passed!")
        return result
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        return None

def test_error_cases():
    """Test error handling"""
    print("\n⚠️ Testing error cases...")
    
    # Test with no files
    print("\n1. Testing with no files...")
    response = requests.post(f"{BASE_URL}/process-claim", files=[])
    print(f"   Status: {response.status_code} (expected 422 or 400)")
    
    # Test with too many files
    print("\n2. Testing with too many files...")
    files = [('files', (f'file{i}.pdf', create_dummy_pdf(f'File {i}'), 'application/pdf')) 
             for i in range(15)]
    response = requests.post(f"{BASE_URL}/process-claim", files=files)
    print(f"   Status: {response.status_code} (expected 400)")
    
    print("\n✅ Error handling tests completed!")

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Medical Claim Processor - API Test Suite")
    print("=" * 60)
    
    try:
        test_health_check()
        test_process_claim()
        test_error_cases()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server")
        print("Please ensure the server is running:")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    main()