import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def upload_pdf(file_path):
    url = "http://localhost:8000/upload_pdf"
    qdrant_url = "http://localhost:6333"
    client = QdrantClient(url=qdrant_url)

    # Create collection with correct vector size

    client.collection_exists(
        collection_name="finance_data"
    )

    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print("✅ Upload successful:")
        print(response.json())
    else:
        print("❌ Upload failed:")
        print(response.status_code)
        print(response.text)

def main():
    # 🔁 Replace this with your actual PDF path
    pdf_path = "C:/Users/navur/gitlab_community/pdf_Data.pdf"
    upload_pdf(pdf_path)

if __name__ == "__main__":
    main()
