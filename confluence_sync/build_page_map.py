from dotenv import load_dotenv
load_dotenv()

import os, json, requests

CONFLUENCE_API_URL = os.getenv("CONFLUENCE_API_URL")
CONFLUENCE_USER = os.getenv("CONFLUENCE_USER")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")

IGNORED_TITLES = {"overview", "getting started in confluence"}

def fetch_personal_space_key():
    url = f"{CONFLUENCE_API_URL}/space"
    res = requests.get(url, auth=(CONFLUENCE_USER, CONFLUENCE_TOKEN))
    res.raise_for_status()
    data = res.json()
    # Look for a space with your name or type 'personal'
    for space in data.get("results", []):
        if space["type"] == "personal":
            return space["key"]
    raise Exception("Personal space key not found")

def fetch_all_pages(space_key):
    pages = []
    start = 0
    limit = 50
    while True:
        url = f"{CONFLUENCE_API_URL}/content"
        params = {"spaceKey": space_key, "type": "page", "expand": "title", "limit": limit, "start": start}
        res = requests.get(url, auth=(CONFLUENCE_USER, CONFLUENCE_TOKEN), params=params)
        res.raise_for_status()
        data = res.json()

        for page in data["results"]:
            title = page["title"].strip().lower()
            if title in IGNORED_TITLES:
                print(f"⏭️ Skipping ignored page: {page['title']}")
                continue
            pages.append((page["title"], page["id"]))

        if len(data["results"]) < limit:
            break
        start += limit
    return dict(pages)

def save_page_map(mapping):
    with open("confluence_sync/page_map.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Saved page_map.json with {len(mapping)} pages")

if __name__ == "__main__":
    try:
        space_key = fetch_personal_space_key()
        page_map = fetch_all_pages(space_key)
        save_page_map(page_map)
    except Exception as e:
        print(f"❌ Failed: {e}")
