import os
import base64
import requests
import sys

GITHUB_USER = "shreya070615"
REPO_NAME = "RBIinsights-rag-pipeline"

def main():
    # Get token from environment or arguments
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        if len(sys.argv) > 1:
            token = sys.argv[1]
        else:
            print("Error: GITHUB_TOKEN environment variable not set, and token not provided as argument.")
            print("Usage: python push_to_github.py <YOUR_GITHUB_TOKEN>")
            sys.exit(1)
            
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 1. Create Repository (or verify if it exists)
    repo_url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}"
    r = requests.get(repo_url, headers=headers)
    if r.status_code == 200:
        print(f"[INFO] Repository '{REPO_NAME}' already exists on GitHub. Proceeding to update files...")
    elif r.status_code == 404:
        print(f"[INFO] Repository '{REPO_NAME}' not found. Creating it...")
        create_url = "https://api.github.com/user/repos"
        payload = {
            "name": REPO_NAME,
            "description": "Compliance Intelligence Platform: Local RAG Pipeline for RBI regulations and custom PDFs.",
            "private": False,
            "auto_init": False
        }
        cr = requests.post(create_url, headers=headers, json=payload)
        if cr.status_code == 201:
            print(f"[SUCCESS] Repository '{REPO_NAME}' created successfully!")
        else:
            print(f"[ERROR] Failed to create repository: {cr.status_code} - {cr.text}")
            sys.exit(1)
    else:
        print(f"[ERROR] Unexpected response checking repository: {r.status_code} - {r.text}")
        sys.exit(1)

    # 2. Get list of files to upload
    files_to_upload = []
    ignored_prefixes = ("venv", "__pycache__", "temp_uploads", ".git", ".idea", ".vscode")
    
    for root, dirs, files in os.walk("."):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if not d.startswith(ignored_prefixes)]
        
        for file in files:
            file_path = os.path.join(root, file)
            # Normalize path representation
            normalized_path = os.path.relpath(file_path, ".").replace("\\", "/")
            
            # Skip ignored file patterns
            if normalized_path.startswith(ignored_prefixes) or normalized_path.endswith((".pyc", ".log", ".png")):
                continue
                
            files_to_upload.append((file_path, normalized_path))
            
    print(f"[INFO] Found {len(files_to_upload)} files to push to GitHub.")
    
    # 3. Upload each file
    for local_path, repo_path in files_to_upload:
        print(f"[UPLOAD] Pushing '{repo_path}'...", end="", flush=True)
        try:
            with open(local_path, "rb") as f:
                content = f.read()
            encoded_content = base64.b64encode(content).decode("utf-8")
            
            # Check if file already exists to get its SHA (for update)
            file_url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/contents/{repo_path}"
            fr = requests.get(file_url, headers=headers)
            
            payload = {
                "message": f"Upload {repo_path} via API",
                "content": encoded_content
            }
            
            if fr.status_code == 200:
                # File exists, update it by providing current SHA
                file_info = fr.json()
                payload["sha"] = file_info["sha"]
                
            # Perform PUT upload
            ur = requests.put(file_url, headers=headers, json=payload)
            if ur.status_code in (200, 201):
                print(" OK")
            else:
                print(f" FAILED ({ur.status_code} - {ur.text[:100]}...)")
        except Exception as e:
            print(f" ERROR: {str(e)}")

    print(f"\n[SUCCESS] Codebase push complete! View it at: https://github.com/{GITHUB_USER}/{REPO_NAME}")

if __name__ == "__main__":
    main()
