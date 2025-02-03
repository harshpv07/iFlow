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


if __name__ == "__main__":
    try:
        pages = fetch_onenote_contents()
        print(pages)
    except Exception as e:
        print(f"An error occurred: {e}")
