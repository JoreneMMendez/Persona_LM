import requests
from bs4 import BeautifulSoup
import time
import random
import csv

class BisayaScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.keywords = ["kumusta", "diay", "paano", "karon", "ngano", "unsay", "gihapon", "dili", "visayas", "cebu"]

    def scrape(self):
        print("Starting...")
        subs = ["r/Visayas", "r/Cebu", "r/Bisaya"]
        all_data = []
        
        for sub in subs:
            url = f"https://old.reddit.com/{sub}/top/?sort=top&t=week&limit=10"
            print(f"Scraping {url}...")
            try:
                res = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(res.text, 'lxml')
                
                # Find all comment bodies
                comments = soup.find_all('div', class_='usertext-body')
                relevant = [c.get_text(strip=True) for c in comments if any(kw in c.get_text().lower() for kw in self.keywords)]
                
                # Pair them up (Parent -> Child)
                for i in range(0, len(relevant)-1, 2):
                    all_data.append((relevant[i], relevant[i+1]))
                    
                print(f"Found {len(relevant)} relevant snippets.")
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(2)
        
        # Save to CSV
        with open("bisaya_data.csv", "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Parent", "Response"])
            for p, r in all_data:
                writer.writerow([p.replace(",", " "), r.replace(",", " ")])
        
        print("Done! Saved to bisaya_data.csv")

BisayaScraper().scrape()
