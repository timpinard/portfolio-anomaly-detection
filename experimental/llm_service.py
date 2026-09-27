#!/usr/bin/env python3
"""LLM service for natural language explanations."""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# LLM client (optional)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = None

if ANTHROPIC_API_KEY:
    try:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized for LLM explanations")
    except ImportError:
        logger.warning("anthropic package not installed. LLM explanations unavailable.")
else:
    logger.warning("ANTHROPIC_API_KEY not set. LLM explanations unavailable.")


def get_llm_client():
    """Get the LLM client if available."""
    return anthropic_client


def is_llm_available() -> bool:
    """Check if LLM service is available."""
    return anthropic_client is not None


def generate_llm_explanation(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2000
) -> str:
    """Generate explanation using LLM."""
    if not anthropic_client:
        raise ValueError("LLM service not configured. Set ANTHROPIC_API_KEY environment variable.")
    
    # Debug: Log the API request
    logger.info(f"LLM API Request:")
    logger.info(f"  Model: {model}")
    logger.info(f"  Max tokens: {max_tokens}")
    logger.info(f"  Prompt length: {len(prompt)} characters")
    logger.debug(f"LLM API Request payload:\n{prompt}")
    
    try:
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Debug: Log the response
        response_text = message.content[0].text
        logger.info(f"LLM API Response:")
        logger.info(f"  Response length: {len(response_text)} characters")
        logger.debug(f"LLM API Response preview: {response_text[:200]}...")
        
        return response_text
    except Exception as e:
        logger.error(f"LLM explanation error: {str(e)}", exc_info=True)
        raise

