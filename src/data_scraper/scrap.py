import argparse
from articles_scrapers import GhostScraper

"""
We use public API, so we don't need to hold it in .env

tags - List of categories tags from https://www.deeplearning.ai/the-batch/ 
that forms endpoint in form:
    https://www.deeplearning.ai/the-batch/tag/{some_tag}/
"""

# --- Argument parser ---
parser = argparse.ArgumentParser(
    description="Scrape articles from The Batch (Ghost CMS) by tags and save to JSON"
)

parser.add_argument(
    "--tags",
    nargs="+",
    default=['letters', 'data-points', 'research', 'business', 'science', 'culture', 'hardware', 'ai-careers'],
    help="List of tags to scrape (space separated). Example: --tags research ai culture"
)

parser.add_argument(
    "--output",
    type=str,
    default="all_articles.json",
    help="Output JSON filename (default: all_articles.json)"
)

parser.add_argument(
    "--api_url",
    type=str,
    default='https://dl-staging-website.ghost.io/ghost/api/content/posts/',
    help="Ghost API URL (default: The Batch staging)"
)

parser.add_argument(
    "--api_key",
    type=str,
    default='a4b216e975091c63cc39c1ac98',
    help="Ghost API Key (default: public key)"
)

args = parser.parse_args()
scraper = GhostScraper(args.api_url, args.api_key)
print(f"\nScraping tags: {args.tags}")
all_posts = scraper.scrape_site(args.tags)
scraper.save_to_json(args.output, all_posts)
