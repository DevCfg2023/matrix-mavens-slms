# matrix-mavens-slms

 pip install -r requirements.txt

# Install Docker
  * Windows: https://www.docker.com/products/docker-desktop/
  * Go to your Start Menu.
  * Find and launch "Docker Desktop".
  * Wait until the Docker whale icon in your system tray shows "Docker is running".
  * docker --version
  * docker pull qdrant/qdrant
  * docker run -p 6333:6333 qdrant/qdrant

  
# Alternative: Run Qdrant without Docker
  * If Docker keeps failing, run Qdrant via binary:
  * Download Qdrant binary from:
  * https://github.com/qdrant/qdrant/releases
  * Extract and run:   ./qdrant
  * It will start at http://localhost:6333

# Run services using below command
 * uvicorn main:app --reload 

 # Refer client
    * Upload
    * fin_chat
