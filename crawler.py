import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =====================================
# Configuration
# =====================================

BASE_URL = "https://api.digikala.com/v1"
OUTPUT_FILE = "digikala_comments.csv"
MAX_WORKERS = 5  # Number of concurrent threads
BATCH_SIZE = 50  # Batch size for saving comments

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7"
}

# Locks for thread-safe operations
file_lock = Lock()
stats_lock = Lock()

# Global statistics
stats = {
    "total_products": 0,
    "processed_products": 0,
    "total_comments": 0,
    "failed_products": 0
    
}

# =====================================
# Session with Retry
# =====================================

def create_session():
    """Create session with automatic retry capability"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# =====================================
# Safe Request
# =====================================

def safe_get(session, url, timeout=15):
    """Make request with error handling"""
    try:
        response = session.get(
            url,
            headers=HEADERS,
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            time.sleep(5)  # Rate limit
            return None
        else:
            return None
            
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None

# =====================================
# Batch Save
# =====================================

def append_to_csv(rows):
    """Save batch with lock"""
    if not rows:
        return
    
    with file_lock:
        df = pd.DataFrame(rows)
        file_exists = os.path.isfile(OUTPUT_FILE)
        
        df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig"
        )

# =====================================
# Extract Products
# =====================================

def get_category_products(session, category_slug, max_pages=20):
    """Extract product list from a category"""
    product_ids = set()
    
    print(f"\nCategory: {category_slug}")
    
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/categories/{category_slug}/search/?page={page}"
        data = safe_get(session, url)
        
        if not data or "data" not in data:
            break
        
        products = data.get("data", {}).get("products", [])
        
        if not products:
            break
        
        for p in products:
            product_ids.add(p["id"])
        
        print(f"  Page {page}: {len(products)} products")
        time.sleep(0.5)
    
    return list(product_ids)

# =====================================
# Extract Comments
# =====================================

def get_product_comments(session, product_id, max_pages=20):
    """Extract comments for a product"""
    comments_data = []
    
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/rate-review/products/{product_id}/?page={page}"
        data = safe_get(session, url)
        
        if not data or "data" not in data:
            break
        
        comments = data.get("data", {}).get("comments", [])
        
        if not comments:
            break
        
        for c in comments:
            comments_data.append({
                "product_id": c.get("product_id"),
                "comment_id": c.get("id"),
                "rating": c.get("rate"),
                "title": c.get("title", ""),
                "comment": c.get("body", ""),
                "likes": c.get("reactions", {}).get("likes", 0),
                "dislikes": c.get("reactions", {}).get("dislikes", 0),
                "is_buyer": c.get("is_buyer", False),
                "created_at": c.get("created_at", "")
            })
        
        # Check if we reached the last page
        pager = data.get("data", {}).get("pager", {})
        if page >= pager.get("total_pages", 1):
            break
        
        time.sleep(0.3)
    
    return comments_data

# =====================================
# Process Single Product
# =====================================

def process_product(product_id):
    """Process a single product in separate thread"""
    session = create_session()
    
    try:
        comments = get_product_comments(session, product_id)
        
        if comments:
            append_to_csv(comments)
        
        with stats_lock:
            stats["processed_products"] += 1
            stats["total_comments"] += len(comments)
        
        return {
            "product_id": product_id,
            "comments_count": len(comments),
            "success": True
        }
        
    except Exception as e:
        with stats_lock:
            stats["failed_products"] += 1
        
        return {
            "product_id": product_id,
            "comments_count": 0,
            "success": False,
            "error": str(e)
        }
    finally:
        session.close()

# =====================================
# Main Execution
# =====================================

def main():
    print("=" * 60)
    print("Starting Digikala Dataset Extraction")
    print("=" * 60)
    
    categories = [
        "mobile-phone",
        "notebook-netbook-ultrabook",
        "tablet",
        "smart-watch",
        "headphone",
        "speaker",
        "game-console",
        "vacuum-cleaner",
        "washing-machine",
        "refrigerator",
        "camera",
        "tv"
    ]
    
    # Phase 1: Collect product list
    print("\nPhase 1: Extracting product list...")
    print("-" * 60)
    
    session = create_session()
    all_products = []
    
    for category in categories:
        products = get_category_products(session, category, max_pages=20)
        all_products.extend(products)
        print(f"  {category}: {len(products)} products\n")
        time.sleep(1)
    
    session.close()
    
    stats["total_products"] = len(all_products)
    
    print(f"\nTotal: {len(all_products)} products found")
    
    # Phase 2: Extract comments with multithreading
    print("\nPhase 2: Extracting comments...")
    print("-" * 60)
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_product, pid): pid 
                for pid in all_products
            }
            
            with tqdm(total=len(all_products), desc="Progress", ncols=80) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    pbar.update(1)
                    
                    if result["success"] and result["comments_count"] > 0:
                        pbar.set_postfix({
                            "comments": stats["total_comments"],
                            "errors": stats["failed_products"]
                        })
    
    except KeyboardInterrupt:
        print("\n\nStopped by user...")
    
    # Final results
    print("\n" + "=" * 60)
    print("Extraction Complete")
    print("=" * 60)
    print(f"Products processed: {stats['processed_products']}/{stats['total_products']}")
    print(f"Total comments: {stats['total_comments']}")
    print(f"Errors: {stats['failed_products']}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
