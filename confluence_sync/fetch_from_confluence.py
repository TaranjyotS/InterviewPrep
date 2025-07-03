from dotenv import load_dotenv
load_dotenv()

import os, json, requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

CONFLUENCE_API_URL = os.getenv("CONFLUENCE_API_URL")
CONFLUENCE_USER = os.getenv("CONFLUENCE_USER")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")
OUT_DIR = "docs"
os.makedirs(OUT_DIR, exist_ok=True)

def get_page_html(page_id):
    url = f"{CONFLUENCE_API_URL}/content/{page_id}?expand=body.storage,title"
    res = requests.get(url, auth=(CONFLUENCE_USER, CONFLUENCE_TOKEN))
    res.raise_for_status()
    data = res.json()
    return data["title"], data["body"]["storage"]["value"]

def convert_html_to_md(html):
    soup = BeautifulSoup(html, "html.parser")
    for macro in soup.find_all("ac:structured-macro"):
        macro.decompose()
    return md(str(soup), heading_style="ATX")

def export_all():
    with open("confluence_sync/page_map.json") as f:
        page_map = json.load(f)

    for title, page_id in page_map.items():
        try:
            clean_title, html = get_page_html(page_id)
            markdown = convert_html_to_md(html)
            file_name = title.lower().replace(" ", "_") + ".md"
            with open(os.path.join(OUT_DIR, file_name), "w", encoding="utf-8") as f:
                f.write(f"# {clean_title}\n\n{markdown}")
            print(f"✅ Exported: {file_name}")
        except Exception as e:
            print(f"❌ Failed: {title} - {e}")

if __name__ == "__main__":
    export_all()
