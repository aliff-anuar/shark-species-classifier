import requests
import os
from PIL import Image
from io import BytesIO
import time

ROOT_DIR = "shark_dataset/raw"

shark_searches = {
    "great_white_shark":   ["great white shark", "Carcharodon carcharias", "white shark"],
    "hammerhead_shark":    ["hammerhead shark", "Sphyrna mokarran", "scalloped hammerhead"],
    "tiger_shark":         ["tiger shark", "Galeocerdo cuvier"],
    "bull_shark":          ["bull shark", "Carcharhinus leucas"],
    "whale_shark":         ["whale shark", "Rhincodon typus"],
    "nurse_shark":         ["nurse shark", "Ginglymostoma cirratum"],
    "mako_shark":          ["mako shark", "Isurus oxyrinchus", "shortfin mako"],
    "blacktip_reef_shark": ["blacktip reef shark", "Carcharhinus melanopterus"],
}

def download_image(url, filepath):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, timeout=15, headers=headers)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(filepath, "JPEG")
        return True
    except:
        return False

def crawl_inaturalist(query, save_path, count, target):
    """Crawl from iNaturalist - free wildlife photo database"""
    page = 1
    while count < target:
        try:
            url = f"https://api.inaturalist.org/v1/observations"
            params = {
                "q": query,
                "photos": "true",
                "per_page": 200,
                "page": page,
                "order_by": "votes",
                "license": "any"
            }
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for obs in results:
                if count >= target:
                    break
                photos = obs.get("photos", [])
                for photo in photos:
                    if count >= target:
                        break
                    img_url = photo.get("url", "").replace("square", "large")
                    if img_url:
                        filepath = os.path.join(save_path, f"{count:06d}.jpg")
                        if download_image(img_url, filepath):
                            count += 1
            page += 1
            time.sleep(1)
        except Exception as e:
            print(f"    iNaturalist error: {e}")
            time.sleep(3)
            break
    return count

def crawl_wikimedia(query, save_path, count, target):
    """Crawl from Wikimedia Commons - completely free"""
    try:
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{query} shark",
            "gsrnamespace": 6,
            "gsrlimit": 500,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 500,
            "format": "json"
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if count >= target:
                break
            imageinfo = page.get("imageinfo", [{}])
            mime = imageinfo[0].get("mime", "")
            if "image" not in mime:
                continue
            img_url = imageinfo[0].get("thumburl") or imageinfo[0].get("url")
            if img_url:
                filepath = os.path.join(save_path, f"{count:06d}.jpg")
                if download_image(img_url, filepath):
                    count += 1
        time.sleep(1)
    except Exception as e:
        print(f"    Wikimedia error: {e}")
    return count

def crawl_unsplash(query, save_path, count, target):
    """Crawl from Unsplash - free stock photos"""
    try:
        for page in range(1, 20):
            if count >= target:
                break
            url = f"https://unsplash.com/napi/search/photos"
            params = {
                "query": query,
                "per_page": 20,
                "page": page
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, params=params, headers=headers, timeout=15)
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for item in results:
                if count >= target:
                    break
                img_url = item.get("urls", {}).get("regular", "")
                if img_url:
                    filepath = os.path.join(save_path, f"{count:06d}.jpg")
                    if download_image(img_url, filepath):
                        count += 1
            time.sleep(0.5)
    except Exception as e:
        print(f"    Unsplash error: {e}")
    return count

# ── Main crawling loop ────────────────────────────────────────
TARGET_PER_CLASS = 1250  # 8 x 1250 = 10,000

for folder_name, keywords in shark_searches.items():
    save_path = os.path.join(ROOT_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)
    count = len(os.listdir(save_path))

    print(f"\n Crawling: {folder_name} (have {count}, need {TARGET_PER_CLASS})")

    for keyword in keywords:
        if count >= TARGET_PER_CLASS:
            break
        print(f"  iNaturalist: '{keyword}'")
        count = crawl_inaturalist(keyword, save_path, count, TARGET_PER_CLASS)
        print(f"    -> {count} images so far")

    for keyword in keywords:
        if count >= TARGET_PER_CLASS:
            break
        print(f"  Wikimedia: '{keyword}'")
        count = crawl_wikimedia(keyword, save_path, count, TARGET_PER_CLASS)
        print(f"    -> {count} images so far")

    for keyword in keywords:
        if count >= TARGET_PER_CLASS:
            break
        print(f"  Unsplash: '{keyword}'")
        count = crawl_unsplash(keyword, save_path, count, TARGET_PER_CLASS)
        print(f"    -> {count} images so far")

    print(f" DONE {folder_name}: {count} images collected")

print(f"\n All done! Check: {ROOT_DIR}")