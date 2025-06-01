import requests
import json
from bs4 import BeautifulSoup
import unicodedata


class GhostScraper:
    def __init__(self, api_url, api_key):
        """Initialize GhostScraper"""
        self.api_url = api_url
        self.api_key = api_key

    def _make_request(self, params):
        """Make GET-request to Ghost API"""
        response = requests.get(self.api_url, params=params)
        response.raise_for_status()
        return response.json()

    def scrape_site(self, tags):
        """
        Scrape entire site by given list of tags.
        Returns list of post dictionaries.
        """
        all_posts = []

        for tag in tags:
            print(f"\n=== Scraping tag: '{tag}' ===")
            page = 1

            while True:
                params = {
                    'key': self.api_key,
                    'limit': 15,
                    'include': 'tags,authors',
                    'filter': f'tag:{tag}',
                    'page': page
                }

                response = self._make_request(params)
                posts_data = response.get('posts', [])

                if not posts_data:
                    print(f"Tag '{tag}', page {page} — no more posts.")
                    break

                print(f"Tag '{tag}', page {page} — found {len(posts_data)} posts.")

                for post_data in posts_data:
                    post_dict = self._post_to_dict(post_data)
                    all_posts.append(post_dict)
                page += 1
        return all_posts

    def _clean_text(self, html_text):
        """Clean HTML text: remove unwanted patterns"""
        soup = BeautifulSoup(html_text or '', 'html.parser')
        clean_text = soup.get_text().strip()

        unwanted_phrases = [
            "Loading the Elevenlabs Text to Speech AudioNative Player..."
        ]

        for phrase in unwanted_phrases:
            if phrase in clean_text:
                clean_text = clean_text.replace(phrase, '')

        clean_text = '\n'.join(line for line in clean_text.splitlines() if line.strip())
        clean_text = unicodedata.normalize("NFKC", clean_text)
        return clean_text

    def _post_to_dict(self, data):
        """Convert post data to dictionary in format: text + metadata (for multimodal RAG)"""
        return {
            'text': self._clean_text(data.get('html')),
            'metadata': {
                'title': data.get('title'),
                'url': data.get('url'),
                'published_at': data.get('published_at'),
                'tags': [tag['name'] for tag in data.get('tags', [])],
                'feature_image': data.get('feature_image')
            }
        }

    def save_to_json(self, filename, data):
        """Save list of posts to one JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"\nSaved {len(data)} posts to {filename}")
