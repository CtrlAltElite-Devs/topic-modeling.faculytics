"""
Discord notification utilities.
"""
import os
import logging
import requests
from dotenv import load_dotenv

from .config import DISCORD_CHANNEL_ID

logger = logging.getLogger(__name__)

# Load .env from repo root
load_dotenv()

DISCORD_API_BASE = "https://discord.com/api/v10"
MAX_MESSAGE_LENGTH = 1900  # Leave buffer under 2000 limit

# Y.A.L.A. bot's own user ID — used to self-ping and trigger next session
YALA_BOT_ID = "1479328192765362317"


def get_bot_token() -> str:
    """Get Discord bot token from environment."""
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN not found in environment. Add it to .env file.")
    return token


def post_to_discord(message: str, channel_id: str = DISCORD_CHANNEL_ID, ping_self: bool = False) -> bool:
    """
    Post a message to Discord channel.
    
    Automatically splits messages over 1900 characters.
    
    Args:
        message: Message content (supports Discord markdown).
        channel_id: Target channel ID.
        
    Returns:
        True if all messages sent successfully.
    """
    token = get_bot_token()
    
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json"
    }
    
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    
    # Split message if too long
    chunks = split_message(message)
    
    success = True
    for i, chunk in enumerate(chunks):
        payload = {"content": chunk}
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Message {i+1}/{len(chunks)} posted to Discord")
            elif response.status_code == 429:
                # Rate limited
                retry_after = response.json().get("retry_after", 5)
                logger.warning(f"Rate limited, waiting {retry_after}s...")
                import time
                time.sleep(retry_after)
                # Retry
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"Retry failed: {response.status_code}")
                    success = False
            else:
                logger.error(f"Discord API error: {response.status_code} - {response.text}")
                success = False
                
        except Exception as e:
            logger.error(f"Failed to post to Discord: {e}")
            success = False
    
    return success


def split_message(message: str) -> list[str]:
    """
    Split a long message into chunks that fit Discord's limit.
    Tries to split on newlines for cleaner breaks.
    """
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]
    
    chunks = []
    current = ""
    
    for line in message.split("\n"):
        # If single line is too long, force split it
        if len(line) > MAX_MESSAGE_LENGTH:
            if current:
                chunks.append(current.strip())
                current = ""
            # Split long line by character
            for i in range(0, len(line), MAX_MESSAGE_LENGTH):
                chunks.append(line[i:i + MAX_MESSAGE_LENGTH])
            continue
        
        # Check if adding this line exceeds limit
        test = current + "\n" + line if current else line
        if len(test) > MAX_MESSAGE_LENGTH:
            chunks.append(current.strip())
            current = line
        else:
            current = test
    
    if current:
        chunks.append(current.strip())
    
    return chunks
