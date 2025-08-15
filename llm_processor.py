"""LLM processor using Ollama."""

import json
import logging
from typing import List, Dict, Any
import requests
from datetime import datetime

from models import ScrapedContent, ProcessedContent, ScrapingResult
from config import Config

class LLMProcessor:
    """Process scraped content using Ollama LLM."""

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ollama_url = f"{self.config.OLLAMA_HOST}/api/generate"

    def process_content_batch(self, contents: List[ScrapedContent]) -> List[ProcessedContent]:
        """Process a batch of scraped content."""
        processed_contents = []

        for content in contents:
            try:
                processed = self._process_single_content(content)
                if processed:
                    processed_contents.append(processed)
            except Exception as e:
                self.logger.error(f"Error processing content {content.url}: {e}")

        return processed_contents

    def _process_single_content(self, content: ScrapedContent) -> ProcessedContent:
        """Process a single piece of content with LLM."""
        prompt = self._create_analysis_prompt(content)

        response = self._call_ollama(prompt)
        if not response:
            return None

        try:
            # Parse LLM response
            analysis = self._parse_llm_response(response)

            return ProcessedContent(
                original=content,
                summary=analysis.get('summary', 'No summary available'),
                relevance_score=analysis.get('relevance_score', 0.5),
                key_topics=analysis.get('key_topics', []),
                urgency_level=analysis.get('urgency_level', 'medium')
            )
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return None

    def _create_analysis_prompt(self, content: ScrapedContent) -> str:
        """Create analysis prompt for the LLM."""
        return f"""Analyze this authentication-related content and provide structured analysis:

Title: {content.title}
Content: {content.content[:1000]}...
Source: {content.source.value}
Author: {content.author}
Timestamp: {content.timestamp}

Please provide analysis in JSON format with:
1. summary: Brief summary of the content (max 200 words)
2. relevance_score: Float between 0-1 indicating relevance to authentication/security
3. key_topics: List of main topics/technologies mentioned
4. urgency_level: "low", "medium", or "high" based on urgency indicators

Respond only with valid JSON:"""

    def _call_ollama(self, prompt: str) -> str:
        """Make request to Ollama API."""
        payload = {
            "model": self.config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result.get('response', '')

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama API request failed: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            return {
                'summary': response[:200],
                'relevance_score': 0.5,
                'key_topics': ['authentication'],
                'urgency_level': 'medium'
            }

    def generate_overall_summary(self, processed_contents: List[ProcessedContent]) -> Dict[str, Any]:
        """Generate overall summary of all processed content."""
        if not processed_contents:
            return {
                'overall_summary': 'No relevant content found.',
                'top_trends': [],
                'high_priority_items': 0
            }

        # Create summary prompt
        summaries = [pc.summary for pc in processed_contents[:10]]  # Limit for prompt size
        prompt = f"""Based on these authentication-related discussions from the last 5 minutes, provide an overall analysis:

Summaries:
{chr(10).join([f"- {s}" for s in summaries])}

Provide JSON response with:
1. overall_summary: Executive summary of current auth trends (max 300 words)
2. top_trends: List of top 5 trending topics/issues
3. high_priority_items: Count of high-urgency items

Respond only with valid JSON:"""

        response = self._call_ollama(prompt)
        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass

        # Fallback
        return {
            'overall_summary': f'Analyzed {len(processed_contents)} authentication-related items.',
            'top_trends': ['authentication', 'security'],
            'high_priority_items': len([pc for pc in processed_contents if pc.urgency_level == 'high'])
        }
