import os
import requests
from datetime import datetime
import msal
import webbrowser

def fetch_onenote_contents():
    """
    Fetches all contents from OneNote notebooks using Microsoft Graph API.
    Returns a list of dictionaries containing page content and metadata.
    """
    # Azure AD app credentials
    client_id = "f832bd00-1e06-4930-9446-8868f7f74201"
    client_secret = "plj8Q~UR-9q-Nj8FyV8qL_X2pPzr9PpTDbGvgcSL"
    tenant_id = "f2b1fc6d-2cfd-4587-98d7-d0b7afd9f0f3"

    app = msal.ConfidentialClientApplication(
        client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
        client_credential=client_secret
    )

    # Acquire a token using client credentials
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])

    if "access_token" not in result:
        raise Exception(f"Failed to obtain access token: {result.get('error_description')}")

    access_token = result['access_token']
    print(access_token)

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    # Get all notebooks using Microsoft Graph API
    graph_api_base = "https://graph.microsoft.com/v1.0"
    # Use /users/{user-id} instead of /me for application permissions
    user_id = "harshpv2@illinois.edu"  # Replace with the actual user ID or email
    notebooks_url = f"{graph_api_base}/users/{user_id}/onenote/notebooks"
    notebooks_response = requests.get(notebooks_url, headers=headers)
    
    # Check if the notebooks request was successful
    if notebooks_response.status_code != 200:
        error_info = notebooks_response.json().get('error', {}).get('message', 'Unknown error')
        raise Exception(f"Failed to fetch notebooks: The request does not contain a valid authentication token. Detailed error information: {error_info}")
    
    notebooks = notebooks_response.json().get('value', [])
    print("notebooks", notebooks)

    all_pages = []

    for notebook in notebooks:
        # Get sections in notebook
        sections_url = f"{graph_api_base}/users/{user_id}/onenote/notebooks/{notebook['id']}/sections"
        sections_response = requests.get(sections_url, headers=headers)
        
        # Check if the sections request was successful
        if sections_response.status_code != 200:
            raise Exception(f"Failed to fetch sections for notebook {notebook['id']}: {sections_response.text}")
        
        sections = sections_response.json().get('value', [])

        for section in sections:
            # Get pages in section
            pages_url = f"{graph_api_base}/users/{user_id}/onenote/sections/{section['id']}/pages"
            pages_response = requests.get(pages_url, headers=headers)
            
            # Check if the pages request was successful
            if pages_response.status_code != 200:
                raise Exception(f"Failed to fetch pages for section {section['id']}: {pages_response.text}")
            
            pages = pages_response.json().get('value', [])

            for page in pages:
                # Get page content
                content_url = f"{graph_api_base}/users/{user_id}/onenote/pages/{page['id']}/content"
                content_response = requests.get(content_url, headers=headers)
                
                # Check if the content request was successful
                if content_response.status_code != 200:
                    raise Exception(f"Failed to fetch content for page {page['id']}: {content_response.text}")
                
                page_data = {
                    'id': page['id'],
                    'title': page['title'],
                    'created_time': page['createdDateTime'],
                    'last_modified': page['lastModifiedDateTime'],
                    'notebook_name': notebook['displayName'],
                    'section_name': section['displayName'],
                    'content': content_response.text
                }
                all_pages.append(page_data)

    return all_pages


def process_pdfs_with_clip(path : str , collection_type : str):
    """Process PDFs from email_examples folder using CLIP embeddings and store in Milvus"""
    import torch
    from transformers import CLIPProcessor, CLIPModel
    import fitz  # PyMuPDF
    import os
    
    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    pdf_dir = path
    if not os.path.isdir(pdf_dir):
        raise ValueError(f"'{pdf_dir}' is not a valid directory")
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory '{pdf_dir}'")
    
    embeddings_data = []
    
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Get CLIP embedding for text
            inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
            text_features = model.get_text_features(**inputs)
            text_embedding = text_features.detach().numpy()[0]
            
            embeddings_data.append({
                'embedding': text_embedding.tolist(),
                'text': text,
                'metadata': {
                    'source': filename,
                    'chunk_index': f"page_{page_num}",
                    'has_images': False
                }
            })
        
        doc.close()
    
    # Store embeddings in Milvus
    store_in_milvus(embeddings_data, collection_type=collection_type)
    
    return len(embeddings_data)


def store_in_milvus(embeddings_data, collection_type):
    """Store embeddings in Milvus collection"""
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    
    dim = len(embeddings_data[0]['embedding'])
    collection_name = f"clip_{collection_type}"

    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields=fields, description=f"CLIP embeddings for {collection_type}")

    # Create or get collection
    if collection_name not in utility.list_collections():
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name="embedding", index_params={"metric_type": "L2"})
    else:
        collection = Collection(collection_name)
        collection.load()

    # Insert data
    data = [
        [e['embedding'] for e in embeddings_data],
        [e['text'] for e in embeddings_data],
        [e['metadata'] for e in embeddings_data]
    ]
    collection.insert(data)
    collection.flush()


if __name__ == "__main__":
    process_pdfs_with_clip("../email_examples", "emails")
    process_pdfs_with_clip("../notes_examples", "notes")
