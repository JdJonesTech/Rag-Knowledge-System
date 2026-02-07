
import json
import logging
import asyncio
import httpx
from bs4 import BeautifulSoup
from pathlib import Path
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.jdjones.com"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "scraped_jd_jones.json"

async def fetch_url(client, url):
    """Fetch content from a URL."""
    try:
        response = await client.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None

def clean_text(text):
    """Clean extracted text."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def extract_content(html, url):
    """Extract relevant content from HTML."""
    soup = BeautifulSoup(html, "lxml")
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.extract()

    # Try to find the main content area
    # This is a heuristic and might need adjustment based on the actual site structure
    # Common classes/ids for content
    content_div = soup.find("main") or \
                 soup.find("div", {"id": "content"}) or \
                 soup.find("div", class_="content") or \
                 soup.find("div", class_="main-content") or \
                 soup.find("body")

    if not content_div:
        return None

    title = soup.title.string if soup.title else ""
    text = clean_text(content_div.get_text(separator="\n"))
    
    # Extract metadata using meta tags if available (description, keywords)
    meta_desc = ""
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag:
        meta_desc = desc_tag.get("content", "")

    return {
        "url": url,
        "title": title,
        "description": meta_desc,
        "content": text
    }

async def scrape_sitemap(client):
    """Fetch and parse the sitemap to get all URLs."""
    xml_content = await fetch_url(client, SITEMAP_URL)
    if not xml_content:
        return []
    
    try:
        root = ET.fromstring(xml_content)
        # Namespace handling for sitemaps usually needed
        # But simple findall with wildcard or namespace might work
        # Sitemap namespace is usually http://www.sitemaps.org/schemas/sitemap/0.9
        
        urls = []
        for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
            loc = url.text
            if loc and loc.startswith(BASE_URL):
                urls.append(loc)
        
        # Fallback if namespace parsing fails or is different
        if not urls:
             for url in root.findall(".//loc"):
                 if url.text.startswith(BASE_URL):
                     urls.append(url.text)
                     
        return list(set(urls)) # Dedup
    except Exception as e:
        logger.error(f"Failed to parse sitemap: {e}")
        return []

async def main():
    logger.info("Starting JD Jones scraper...")
    
    async with httpx.AsyncClient(headers={"User-Agent": "JDJonesBot/1.0"}) as client:
        # 1. Get URLs from sitemap
        logger.info(f"Fetching sitemap from {SITEMAP_URL}...")
        urls = await scrape_sitemap(client)
        logger.info(f"Found {len(urls)} URLs in sitemap.")
        
        # Filter URLs - basic filtering
        # We might want to exclude admin links or irrelevant files (images, pdfs if we don't handle them here)
        # The sitemap snippet showed some .pdf links? No, they were page links.
        # But let's filter out non-html extensions just in case, unless we want to handle them.
        target_urls = [u for u in urls if not u.endswith(('.jpg', '.png', '.pdf', '.xml'))]
        
        # Limit for safety during initial run/testing? User asked for EVERY POSSIBLE info.
        # But fetching too many might be slow.
        # Let's do it in batches or just go for it but with a concurrency limit.
        
        scraped_data = []
        
        # Process URLs
        sem = asyncio.Semaphore(5) # Concurrency limit
        
        async def process_url(url):
            async with sem:
                logger.info(f"Scraping {url}...")
                html = await fetch_url(client, url)
                if html:
                    data = extract_content(html, url)
                    if data and len(data['content']) > 100: # Filter empty pages
                        return data
                return None

        tasks = [process_url(url) for url in target_urls]
        results = await asyncio.gather(*tasks)
        
        for res in results:
            if res:
                # Format for ingestion
                # We need to map this to the structure ingest_data expects OR
                # save it as raw scraping data and let ingest_data transform it.
                # The task said "save output ... in a format compatible with ingest_data.py"
                # ingest_data expects:
                # { "id": ..., "content": ..., "source": ..., "category": ... }
                
                doc_id = res['url'].replace(BASE_URL, "").strip("/").replace("/", "_").replace("-", "_")
                if not doc_id: doc_id = "home"
                
                doc = {
                    "id": f"scraped_{doc_id}",
                    "content": f"Title: {res['title']}\nDescription: {res['description']}\nURL: {res['url']}\n\n{res['content']}",
                    "source": res['url'],
                    "category": "scraped_web_content",
                    "access_level": "public"
                }
                scraped_data.append(doc)

        logger.info(f"Scraped {len(scraped_data)} valid pages.")
        
        # Save to file
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, indent=2)
        
        logger.info(f"Saved data to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
