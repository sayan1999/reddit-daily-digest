import praw
import time
from dotenv import load_dotenv
import os
import sqlite3, requests
import datetime
from litellm import completion
import re
from argparse import ArgumentParser
import logging
from concurrent.futures import ThreadPoolExecutor

os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data/reddit_scraper.log"), logging.StreamHandler()],
)


def get_prompt(type):
    return open("prompt.txt", "r").read().format(type=type)


def init_db():
    conn = sqlite3.connect("data/reddit_posts.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            timestamp TEXT,
            date TEXT,
            type TEXT,
            content TEXT,
            summary TEXT,
            PRIMARY KEY (date, type)
        )
        """
    )
    conn.commit()
    return conn


def if_exists(conn, date, type):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) FROM posts WHERE date = ? AND type = ?
        """,
        (date, type),
    )
    return cursor.fetchone()[0] > 0


def insert_post(conn, timestamp, date, type, content, summary):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO posts (timestamp, date, type, content, summary)
        VALUES (?, ?, ?, ?, ?)
        """,
        (timestamp, date, type, content, summary),
    )
    conn.commit()


def get_summary(content):
    response = completion(
        model="gemini/gemini-2.5-flash-preview-04-17",
        messages=[{"role": "user", "content": get_prompt(type) + "\n\n\n" + content}],
        fallbacks=["gemini/gemini-2.0-flash"],
        num_retries=10,
    )
    return (
        response["choices"][0]["message"]["content"]
        .strip("```markdown")
        .strip("```")
        .strip()
    )


def send_discord_channel(content, type):
    url = os.getenv(f"{type.lower()}_webhook")
    data = {
        "content": content[:2000],
        "username": "Reddit Digest",
    }

    result = requests.post(url, json=data)
    result.raise_for_status()


def reddit_scrape(post_url, max_root_comments=100, max_depth_per_comment=20):
    try:
        if "/comments/" not in post_url:
            logging.warning("Invalid Reddit post URL. %s", post_url)
            return ""
        logging.info(f"Scraping {post_url}...")
        submission = reddit.submission(id=post_url.split("/comments/")[1].split("/")[0])
        submission.comments.replace_more(limit=max_depth_per_comment)
        root_comments = list(submission.comments)[:max_root_comments]
    except:
        logging.error("Failed to scrape Reddit post %s", post_url, exc_info=True)
        return ""

    def process_comments(comments, level=0):
        if level >= max_depth_per_comment:
            return []
        return [
            f"{'    ' * level}COMMENT by u/{comment.author.name if comment.author else '[deleted]'} (Score: {comment.score}):\n{'    ' * level}{comment.body}\n"
            + "".join(process_comments(comment.replies, level + 1))
            for comment in comments
        ]

    return "\n".join(
        [
            f"TITLE: {submission.title}",
            f"AUTHOR: u/{submission.author.name if submission.author else '[deleted]'}",
            f"SCORE: {submission.score}",
            f"POST CONTENT:\n{submission.selftext}\n",
            f"URL: {submission.url}",
        ]
        + process_comments(root_comments)
    )


def filter_by_date(submission, target_date):
    submission_date = datetime.datetime.fromtimestamp(submission.created_utc).date()
    return submission_date == target_date


def chunk_markdown(text, max_chunk_size=2000):
    chunks = []
    current_chunk = ""

    lines = text.split("\n")

    # Process each line
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a header
        if line.startswith("#"):
            if current_chunk and len(current_chunk + line) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
            i += 1
            continue

        # Check if this is a bullet point
        if line.strip().startswith("-"):
            # Collect the entire bullet point including any following indented content
            bullet_point = line + "\n"
            i += 1

            # Look ahead for posts or other related content
            while i < len(lines) and (
                not lines[i].strip().startswith("-") and not lines[i].startswith("#")
            ):
                bullet_point += lines[i] + "\n"
                i += 1

            # Check if adding this bullet point would exceed chunk size
            if (
                len(current_chunk) + len(bullet_point) > max_chunk_size
                and current_chunk
            ):
                chunks.append(current_chunk.strip())
                current_chunk = bullet_point
            else:
                current_chunk += bullet_point
        else:
            # Handle other lines
            if len(current_chunk) + len(line) + 1 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
            i += 1

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


wrap_links = lambda text: re.sub(r"\(((https?://)[^\s)]+)\)", r"(<\1>)", text)


if __name__ == "__main__":

    parser = ArgumentParser(description="Reddit Scraper")
    parser.add_argument(
        "--cron_job",
        action="store_true",
        help="Run the script in cron job mode (one run only)",
    )
    args = parser.parse_args()
    logging.info("Starting Reddit Scraper...")
    load_dotenv()
    conn = init_db()
    subreddits = {
        "Stock": [
            "IndianStockMarket",
            "MutualfundsIndia",
            "IndiaInvestments",
            "personalfinanceindia",
            "IndianStreetBets",
        ],
        "LLM": ["LocalLLaMA", "OpenAI", "ChatGPT", "Bard"],
        "AI_Art": ["comfyui", "StableDiffusion", "aivideo", "aiVideoCraft"],
        "AI": [
            "MachineLearning",
            "ArtificialInteligence",
            "singularity",
            "learnmachinelearning",
        ],
    }

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    while True:

        today = datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        )

        target_date = (today - datetime.timedelta(days=1)).date()

        for type, subreddits_list in subreddits.items():
            if if_exists(conn, str(target_date), type):
                logging.info(f"Already scraped {type} for {target_date}")
                continue
            content = ""
            for subreddit in subreddits_list:
                logging.info(f"Scraping {type} from r/{subreddit}...")
                for category in ["hot"]:
                    submissions = (
                        getattr(reddit.subreddit(subreddit), category)(limit=1000)
                        if category != "top"
                        else getattr(reddit.subreddit(subreddit), category)(
                            limit=1000, time_filter="day"
                        )
                    )

                    def process_submission(submission):
                        if filter_by_date(submission, target_date):
                            logging.info(
                                f"Found post on {target_date}: {submission.title} {submission.url}"
                            )
                            return reddit_scrape(submission.url)
                        else:
                            logging.warning(f"Post out of date {target_date}")
                            return ""

                    with ThreadPoolExecutor(max_workers=10) as executor:
                        results = list(executor.map(process_submission, submissions))

                    content += "\n\n".join(filter(None, results))
            summary = get_summary(content)
            logging.debug(f"Summary for {type}: {summary}")
            send_discord_channel("# " + type, type)
            for summary_chunk in chunk_markdown(summary):
                summary_chunk = wrap_links(summary_chunk)
                logging.debug(f"Sending chunk to Discord: {type=} {summary_chunk}")
                send_discord_channel(summary_chunk, type)
            insert_post(conn, str(today), str(target_date), type, content, summary)
            time.sleep(10)
        if args.cron_job:
            logging.info("Running in cron job mode, exiting after one run.")
            break
        logging.info("Sleeping for 1 hour...")
        time.sleep(3600)
