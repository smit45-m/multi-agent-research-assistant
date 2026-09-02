"""
Integration tests for document upload and management API endpoints.
"""
import pytest
import io

def test_upload_document(test_client):
    file_content = b"This is a test document about research assistants."
    file = io.BytesIO(file_content)
    
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", file, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["status"] == "processed"

def test_list_documents(test_client):
    response = test_client.get("/api/v1/documents/")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)

def test_upload_unsupported_format(test_client):
    file_content = b"Bad format data"
    file = io.BytesIO(file_content)
    
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.unsupported_format_xyz", file, "application/octet-stream")}
    )
    assert response.status_code == 400
