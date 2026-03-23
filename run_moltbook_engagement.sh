#!/bin/bash

# Clawbot Moltbook Engagement - Cron Wrapper
# Runs every 30 minutes to autonomously engage with Moltbook

set -e

# Configuration
CLAWBOT_DIR="/home/pradeep/Downloads/clawbot"
LOG_FILE="/tmp/clawbot-engagement.log"
VENV_PYTHON="${CLAWBOT_DIR}/.venv/bin/python"

# Ensure we're in the right directory
cd "$CLAWBOT_DIR"

# Load environment from .env
export $(grep -v '^#' .env | xargs)

# Run engagement with logging
{
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting Clawbot Moltbook engagement"

    if [ ! -f "$VENV_PYTHON" ]; then
        echo "ERROR: Virtual environment not found at $VENV_PYTHON"
        exit 1
    fi

    # Run heartbeat: check 10 posts, respond to relevant ones
    # With retry on timeout (Moltbook can be slow)
    RETRY_COUNT=0
    MAX_RETRIES=2

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if "$VENV_PYTHON" moltbook_heartbeat.py --limit 10; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Engagement complete"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') - Engagement failed, retrying in 10 seconds... (attempt $RETRY_COUNT of $MAX_RETRIES)"
                sleep 10
            else
                echo "$(date '+%Y-%m-%d %H:%M:%S') - Engagement failed after $MAX_RETRIES attempts"
            fi
        fi
    done
} >> "$LOG_FILE" 2>&1

echo "✅ Cron job ran at $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
