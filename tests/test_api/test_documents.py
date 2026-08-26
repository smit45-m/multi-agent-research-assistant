import pytest
import io
from unittest.mock import patch

@pytest.mark.asyncio
async def test_upload_document(test_client):
    file_content = b"This is a test document."
    file = io.BytesIO(file_content)
    file.name = "test.txt"
    
    with patch("app.main.vector_store.add_documents") as mock_add:
        mock_add.return_value = None
        response = await test_client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", file, "text/plain")}
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_list_documents(test_client):
    response = await test_client.get("/api/v1/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_upload_unsupported_format(test_client):
    file_content = b"Bad format data"
    file = io.BytesIO(file_content)
    file.name = "test.xyz"
    
    response = await test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.xyz", file, "application/octet-stream")}
    )
    assert response.status_code == 400
