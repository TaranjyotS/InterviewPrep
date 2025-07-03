from dotenv import load_dotenv
load_dotenv()

import os, sys, json, requests
from markdown import markdown

CONFLUENCE_API_URL = os.getenv("CONFLUENCE_API_URL")
CONFLUENCE_USER = os.getenv("CONFLUENCE_USER")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")

def get_page_id(title):
    with open("confluence_sync/page_map.json", "r") as f:
        mapping = json.load(f)
    return mapping.get(title)

def get_page_version(page_id):
    res = requests.get(
        f"{CONFLUENCE_API_URL}/content/{page_id}?expand=version",
        auth=(CONFLUENCE_USER, CONFLUENCE_TOKEN)
    )
    res.raise_for_status()
    return res.json()["version"]["number"]

def update_confluence_page(title, md_path):
    page_id = get_page_id(title)
    if not page_id:
        raise ValueError(f"Page ID not found for title '{title}'")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    html_body = markdown(md_text)

    version = get_page_version(page_id)

    payload = {
        "version": {"number": version + 1},
        "title": title,
        "type": "page",
        "body": {
            "storage": {
                "value": html_body,
                "representation": "storage"
            }
        }
    }

    res = requests.put(
        f"{CONFLUENCE_API_URL}/content/{page_id}",
        json=payload,
        auth=(CONFLUENCE_USER, CONFLUENCE_TOKEN),
        headers={"Content-Type": "application/json"}
    )
    res.raise_for_status()
    print(f"✅ Updated '{title}' successfully")

if __name__ == "__main__":
    file_path = sys.argv[1]
    title = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()
    update_confluence_page(title, file_path)
