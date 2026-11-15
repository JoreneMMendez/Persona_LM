import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import random
import os
import re

class BisayaRedditScraper:
    def __init__(self, output_file="bisaya_chats.csv"):
        self.output_file = output_file
        self.data_collected = []
        
        # Bisaya/Bisayaglish keywords for filtering
        self.keywords = [
            "kumusta", "diay", "paano", "karon", "ngano", "unsay", 
            "gihapon", "dili", "visayas", "cebu", "bohol", "negros", 
            "iloilo", "salamat", "uy", "eh", "hala", "hay", "oy",
            "ako", "ikaw", "kami", "nato", "nyo", "kaon", "luto",
            "bisan", "kaya", "dili", "sige", "tambay", "panabi"
        ]
        
        # Subreddits to target
        self.subreddits = ["r/Visayas", "r/Cebu", "r/Bisaya", "r/AskCebuano", "r/Philippines"]
        
        # Headers for requests
        self.headers = {
            'User-Agent': 'BisayaAI_Trainer/1.0 by /u/BisayaBot',
            'Accept': 'application/json, text/html, application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }

    def is_bisaya(self, text):
        """Checks if text contains Bisaya indicators."""
        if not text or len(text) < 5:
            return False
        text_lower = text.lower()
        score = sum(1 for kw in self.keywords if kw in text_lower)
        # Require at least 2 keyword matches to avoid noise
        return score >= 2

    def extract_comments_from_json(self, post_data):
        """Extracts comments from a JSON post object."""
        comments = []
        try:
            # Reddit JSON structure: data -> children -> data -> body
            # We need to traverse the comment tree
            def traverse(node, parent_text=None):
                if isinstance(node, dict) and 'data' in node:
                    data = node['data']
                    body = data.get('body', '')
                    
                    if self.is_bisaya(body):
                        if parent_text:
                            comments.append((parent_text, body))
                        else:
                            # If no parent, we just store it as a standalone for now
                            # or pair with the next relevant comment later
                            pass
                    
                    replies = data.get('replies')
                    if isinstance(replies, dict) and 'children' in replies:
                        for child in replies['children']:
                            traverse(child, body)
                            
                elif isinstance(node, list):
                    for item in node:
                        traverse(item, parent_text)
                        
                elif isinstance(node, str):
                    # Handle cases where replies are just a string "more..."
                    pass

            # Get all comments from the post
            if 'comments' in post_data:
                traverse(post_data['comments'])
                
        except Exception as e:
            print(f"Error parsing JSON comments: {e}")
            
        return comments

    def scrape_subreddit_api(self, subreddit_name):
        """Method 1: Uses the official JSON endpoint."""
        url = f"https://www.reddit.com/{subreddit_name}/top.json?sort=top&t=week&limit=25"
        print(f"[API] Fetching {url}...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                all_comments = []
                
                for post in posts:
                    post_data = post.get('data', {})
                    # Extract comments from each post
                    comments = self.extract_comments_from_json(post_data)
                    all_comments.extend(comments)
                    
                return all_comments
            else:
                print(f"[API] Error {response.status_code}: {response.text[:100]}")
                return []
        except Exception as e:
            print(f"[API] Exception: {e}")
            return []

    def scrape_subreddit_html(self, subreddit_name):
        """Method 2: Fallback HTML scraping using BeautifulSoup."""
        url = f"https://old.reddit.com/{subreddit_name}/top/?sort=top&t=week"
        print(f"[HTML] Fetching {url}...")
        
        try:
            response = requests.get(url, headers={**self.headers, 'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                comments = []
                
                # Find all comment containers
                # Note: This is a simplified extraction; Reddit's HTML is complex
                comment_blocks = soup.find_all('div', class_=lambda x: x and 'comment' in x.lower())
                
                current_parent = None
                for block in comment_blocks:
                    text_el = block.find('div', class_='usertext-body') or block.find('div', class_='md')
                    if text_el:
                        text = text_el.get_text(strip=True)
                        if self.is_bisaya(text):
                            if current_parent:
                                comments.append((current_parent, text))
                                current_parent = None
                            else:
                                current_parent = text
                
                return comments
            else:
                print(f"[HTML] Error {response.status_code}")
                return []
        except Exception as e:
            print(f"[HTML] Exception: {e}")
            return []

    def run(self):
        print("Starting Bisaya Scraper with Multiple Solutions...")
        
        for sub in self.subreddits:
            print(f"\n--- Processing {sub} ---")
            
            # Try API first
            api_data = self.scrape_subreddit_api(sub)
            if not api_data:
                print("[API] Failed. Trying HTML fallback...")
                html_data = self.scrape_subreddit_html(sub)
                if html_data:
                    self.data_collected.extend(html_data)
                    print(f"[HTML] Found {len(html_data)} conversations.")
                else:
                    print("[HTML] Also failed.")
            else:
                self.data_collected.extend(api_data)
                print(f"[API] Found {len(api_data)} conversations.")
            
            # Rate limiting
            time.sleep(random.uniform(1, 3))

        self.save_to_csv()

    def save_to_csv(self):
        if not self.data_collected:
            print("No data collected.")
            return
            
        filename = self.output_file
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Parent_Comment", "Response"])
            
            # Clean and write data
            for parent, response in self.data_collected:
                clean_parent = parent.replace('\n', ' ').replace(',', ' ')
                clean_response = response.replace('\n', ' ').replace(',', ' ')
                writer.writerow([clean_parent, clean_response])
                
        print(f"\nSaved {len(self.data_collected)} rows to {filename}")

if __name__ == "__main__":
    scraper = BisayaRedditScraper(output_file="bisaya_chats.csv")
    scraper.run()