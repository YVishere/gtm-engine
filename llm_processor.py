"""LLM processor using Ollama."""

import json
import logging
from typing import List, Dict, Any
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from models import ScrapedContent, ProcessedContent, ScrapingResult
from config import Config

class LLMProcessor:
    """Process scraped content using Ollama LLM with smart batching and progress tracking."""

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ollama_url = f"{self.config.OLLAMA_HOST}/api/generate"
        
        # Smart batching configuration
        self.BATCH_SIZE = 10  # Process items in batches for better progress tracking
        self.MAX_PARALLEL_REQUESTS = 3  # Limit concurrent API calls
        self.RELEVANCE_THRESHOLD = 0.3  # Lower threshold to see more results (was 0.7)
        self.MAX_ITEMS_TO_PROCESS = 1000  # Cap for production performance

    def process_content_batch(self, contents: List[ScrapedContent]) -> List[ProcessedContent]:
        """Process a batch of scraped content with smart filtering and progress tracking."""
        self.logger.info(f"Starting processing of {len(contents)} items")
        
        # Step 1: Smart pre-filtering using keyword density
        filtered_contents = self._smart_prefilter(contents)
        self.logger.info(f"Pre-filtered to {len(filtered_contents)} high-potential items")
        
        # Step 2: Quick relevance scoring for all items
        quick_scored = self._quick_relevance_batch(filtered_contents)
        
        # Step 3: Sort by relevance and limit for deep processing
        high_relevance = [item for item in quick_scored if item['relevance'] >= self.RELEVANCE_THRESHOLD]
        high_relevance.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Limit processing for performance
        items_to_process = high_relevance[:self.MAX_ITEMS_TO_PROCESS]
        self.logger.info(f"Selected {len(items_to_process)} items for deep LLM analysis")
        
        # Step 4: Deep LLM processing with progress bar
        processed_contents = self._deep_process_with_progress(items_to_process)
        
        self.logger.info(f"Successfully processed {len(processed_contents)} items")
        return processed_contents

    def _smart_prefilter(self, contents: List[ScrapedContent]) -> List[ScrapedContent]:
        """Smart pre-filtering based on keyword density and metadata."""
        filtered = []
        
        for content in contents:
            # Calculate keyword density
            text = f"{content.title} {content.content}".lower()
            keyword_hits = sum(1 for keyword in self.config.AUTH_KEYWORDS if keyword in text)
            
            # Score based on multiple factors
            score = 0
            score += keyword_hits * 10  # Keyword density
            score += min(content.score, 50)  # Reddit/SO score (capped)
            score += min(content.comments_count, 20)  # Engagement
            
            # Quality filters
            if len(content.title) < 10:  # Too short
                score -= 20
            if len(content.content) < 50:  # No substantial content
                score -= 10
            
            # Keep items with reasonable potential
            if score >= 15:
                filtered.append(content)
        
        return filtered

    def _quick_relevance_batch(self, contents: List[ScrapedContent]) -> List[Dict]:
        """Quick relevance scoring using simple heuristics."""
        scored_items = []
        
        print(f"📊 Quick relevance scoring for {len(contents)} items...")
        
        for i, content in enumerate(contents):
            if i % 50 == 0:  # Progress every 50 items
                print(f"   Analyzed {i}/{len(contents)} items", end='\r')
            
            relevance = self._calculate_quick_relevance(content)
            scored_items.append({
                'content': content,
                'relevance': relevance
            })
        
        print(f"   Completed relevance scoring for {len(contents)} items ✅")
        return scored_items

    def _calculate_quick_relevance(self, content: ScrapedContent) -> float:
        """Calculate quick relevance score without LLM."""
        text = f"{content.title} {content.content}".lower()
        
        # High-value keywords (weighted)
        high_value_keywords = {
            'jwt': 0.2, 'oauth': 0.2, 'authentication': 0.15, 'saml': 0.15,
            'sso': 0.15, 'security': 0.1, 'login': 0.1, 'session': 0.1,
            'authorize': 0.1, 'credential': 0.1, 'token': 0.08
        }
        
        score = 0.0
        for keyword, weight in high_value_keywords.items():
            if keyword in text:
                # Count occurrences with diminishing returns
                count = text.count(keyword)
                score += weight * min(count, 3) / 3
        
        # Boost for question formats (indicates real problems)
        if any(word in text for word in ['how', 'why', 'what', 'help', 'issue', 'problem', 'error']):
            score += 0.1
        
        # Boost for implementation discussions
        if any(word in text for word in ['implement', 'integration', 'setup', 'configure']):
            score += 0.1
        
        return min(score, 1.0)

    def _deep_process_with_progress(self, scored_items: List[Dict]) -> List[ProcessedContent]:
        """Deep LLM processing with progress tracking and batching."""
        processed_contents = []
        total_items = len(scored_items)
        
        print(f"\n🤖 Deep LLM analysis of top {total_items} items...")
        print("=" * 60)
        
        # Process in batches with progress
        for batch_start in range(0, total_items, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, total_items)
            batch = scored_items[batch_start:batch_end]
            
            print(f"📦 Processing batch {batch_start//self.BATCH_SIZE + 1}/{(total_items-1)//self.BATCH_SIZE + 1} "
                  f"(items {batch_start+1}-{batch_end})")
            
            # Process batch with threading for I/O efficiency
            batch_results = self._process_batch_parallel(batch)
            processed_contents.extend(batch_results)
            
            # Progress update
            processed_count = len(processed_contents)
            success_rate = (processed_count / (batch_end)) * 100 if batch_end > 0 else 0
            print(f"   ✅ Completed: {processed_count}/{batch_end} items ({success_rate:.1f}% success rate)")
            
            # Brief pause to avoid overwhelming the API
            if batch_end < total_items:
                time.sleep(0.5)
        
        print(f"\n🎉 Deep analysis complete! Processed {len(processed_contents)} items successfully")
        return processed_contents

    def _process_batch_parallel(self, batch: List[Dict]) -> List[ProcessedContent]:
        """Process a batch of items in parallel with controlled concurrency."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_REQUESTS) as executor:
            # Submit all tasks
            future_to_content = {
                executor.submit(self._process_single_content_safe, item['content']): item
                for item in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_content):
                try:
                    result = future.result(timeout=60)  # 60s timeout per item
                    if result:
                        results.append(result)
                except Exception as e:
                    content = future_to_content[future]['content']
                    self.logger.warning(f"Failed to process item '{content.title[:50]}...': {e}")
        
        return results

    def _process_single_content_safe(self, content: ScrapedContent) -> ProcessedContent:
        """Safe wrapper for single content processing with error handling and validation."""
        try:
            result = self._process_single_content(content)
            
            # Validate result has all required fields
            if result and self._validate_processed_content(result):
                return result
            else:
                self.logger.warning(f"Invalid LLM result for '{content.title[:50]}...', creating fallback")
                return self._create_fallback_processed_content(content)
                
        except Exception as e:
            self.logger.error(f"Error processing content {content.url}: {e}")
            return self._create_fallback_processed_content(content)
    
    def _validate_processed_content(self, content: ProcessedContent) -> bool:
        """Validate that processed content has all required fields."""
        if not content:
            return False
        
        # Check required fields
        required_checks = [
            content.summary and len(content.summary.strip()) > 10,
            isinstance(content.relevance_score, (int, float)) and 0 <= content.relevance_score <= 1,
            content.key_topics and len(content.key_topics) > 0,
            content.urgency_level in ['low', 'medium', 'high']
        ]
        
        return all(required_checks)
    
    def _create_fallback_processed_content(self, content: ScrapedContent) -> ProcessedContent:
        """Create fallback processed content when LLM fails."""
        # Extract basic info from title and content
        text = f"{content.title} {content.content}".lower()
        
        # Simple relevance calculation
        relevance = self._calculate_quick_relevance(content)
        
        # Extract topics from keywords
        topics = []
        for keyword in self.config.AUTH_KEYWORDS:
            if keyword in text:
                topics.append(keyword)
        topics = topics[:4] if topics else ['authentication']
        
        # Determine urgency from context
        urgency = 'high' if any(word in text for word in ['urgent', 'help', 'error', 'broken', 'issue']) else 'medium'
        
        # Create basic summary
        summary = f"Discussion about {', '.join(topics[:2])} in {content.source.value}. " \
                 f"Author {content.author} seeking guidance on authentication implementation."
        
        return ProcessedContent(
            original=content,
            summary=summary,
            relevance_score=max(relevance, 0.3),  # Ensure minimum relevance
            key_topics=topics,
            urgency_level=urgency
        )

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

    def generate_opportunity_explanation(self, content: ProcessedContent) -> str:
        """Generate GTM-focused explanation for why this is a high-value opportunity."""
        # Use topic-based smart fallback for consistency
        return self._fallback_explanation(content)
    
    def _clean_gtm_response(self, response: str) -> str:
        """Clean and format GTM response."""
        explanation = response.strip()
        
        # Remove thinking sections for DeepSeek
        if '</thinking>' in explanation:
            explanation = explanation.split('</thinking>')[-1].strip()
        
        # Remove JSON/formatting artifacts
        explanation = explanation.replace('```', '').replace('"', '').replace('{', '').replace('}', '').strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "This represents", "This is a", "This discussion", 
            "The opportunity", "Business opportunity:", "GTM opportunity:"
        ]
        for prefix in prefixes_to_remove:
            if explanation.lower().startswith(prefix.lower()):
                explanation = explanation[len(prefix):].strip()
        
        # Ensure proper capitalization
        if explanation and not explanation[0].isupper():
            explanation = explanation[0].upper() + explanation[1:]
        
        # Limit length
        if len(explanation) > 180:
            sentences = explanation.split('. ')
            explanation = '. '.join(sentences[:2])
            if not explanation.endswith('.'):
                explanation += '.'
        
        return explanation
    
    def _fallback_explanation(self, content: ProcessedContent) -> str:
        """Generate fallback GTM explanation with business context."""
        topic_focus = content.key_topics[0] if content.key_topics else "authentication"
        
        # Enhanced business context based on topics
        if any(term in topic_focus.lower() for term in ['oauth', 'jwt', 'saml', 'sso']):
            return f"Developer implementing {topic_focus} authentication presents qualified lead opportunity. " \
                   f"Sales should offer technical consultation showcasing Descope's enterprise auth solutions."
        
        elif any(term in topic_focus.lower() for term in ['api', 'integration', 'setup']):
            return f"API authentication challenges indicate timing opportunity for auth solution evaluation. " \
                   f"Marketing should provide technical resources and schedule discovery call."
        
        elif any(term in topic_focus.lower() for term in ['security', 'vulnerability', 'breach']):
            return f"Security-focused discussion signals urgent need for robust authentication. " \
                   f"Sales should prioritize immediate outreach with security-first messaging."
        
        else:
            urgency_map = {"high": "urgent", "medium": "qualified", "low": "potential"}
            urgency_desc = urgency_map.get(content.urgency_level, "potential")
            
            return f"Developer discussing {topic_focus} presents {urgency_desc} lead opportunity. " \
                   f"Team should engage with technical consultation and Descope platform demo."

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

IMPORTANT: Respond ONLY with valid JSON. Do not include thinking steps or explanations. Example format:
{{
    "summary": "Brief content summary here",
    "relevance_score": 0.8,
    "key_topics": ["jwt", "oauth", "security"],
    "urgency_level": "medium"
}}"""

    def _call_ollama(self, prompt: str) -> str:
        """Make request to Ollama API."""
        # Get model-specific configuration
        model_config = self.config.MODEL_CONFIGS.get(
            self.config.OLLAMA_MODEL, 
            self.config.MODEL_CONFIGS["llama3.2:1b"]  # fallback
        )
        
        payload = {
            "model": self.config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        
        # Only add format if the model supports it
        if model_config.get("format"):
            payload["format"] = model_config["format"]

        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=model_config.get("timeout", 30)
            )
            response.raise_for_status()

            result = response.json()
            return result.get('response', '')

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama API request failed: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response, handling DeepSeek R1 thinking format."""
        try:
            # First try direct JSON parsing (for llama3.2:1b)
            result = json.loads(response)
            self.logger.debug("Parsed response using direct JSON")
            return result
        except json.JSONDecodeError:
            # Handle DeepSeek R1 format with thinking sections
            try:
                # DeepSeek R1 often outputs thinking + actual response
                # Look for JSON after </thinking> tag or in the final output
                if '</thinking>' in response:
                    # Extract content after thinking section
                    actual_response = response.split('</thinking>')[-1].strip()
                    result = json.loads(actual_response)
                    self.logger.debug("Parsed response after </thinking> tag")
                    return result
                elif '```json' in response:
                    # Extract JSON from code block
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    json_content = response[json_start:json_end].strip()
                    result = json.loads(json_content)
                    self.logger.debug("Parsed response from JSON code block")
                    return result
                else:
                    # Try to find JSON-like content in the response
                    import re
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, response)
                    if matches:
                        result = json.loads(matches[-1])  # Use the last/most complete JSON
                        self.logger.debug("Parsed response using regex JSON extraction")
                        return result
            except (json.JSONDecodeError, IndexError):
                pass
            
            # Fallback parsing if JSON is malformed
            self.logger.warning("Using fallback parsing for malformed LLM response")
            return {
                'summary': response[:200],
                'relevance_score': 0.5,
                'key_topics': ['authentication'],
                'urgency_level': 'medium'
            }

    def generate_overall_summary(self, processed_contents: List[ProcessedContent]) -> Dict[str, Any]:
        """Generate overall summary of all processed content with smart aggregation."""
        if not processed_contents:
            return {
                'overall_summary': 'No relevant content found.',
                'top_trends': [],
                'high_priority_items': 0
            }

        print(f"\n📈 Generating executive summary from {len(processed_contents)} analyzed items...")

        # Smart summary generation using top items only
        top_items = sorted(processed_contents, key=lambda x: x.relevance_score, reverse=True)[:10]
        summaries = [pc.summary for pc in top_items]
        
        # Aggregate topics for trend analysis
        all_topics = []
        for pc in processed_contents:
            all_topics.extend(pc.key_topics)
        
        # Count topic frequency for trends
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        top_trends = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        prompt = f"""Based on these top authentication-related discussions from recent analysis, provide an executive summary:

High-Value Summaries:
{chr(10).join([f"- {s}" for s in summaries])}

Analytics:
- Total items analyzed: {len(processed_contents)}
- High-priority items: {len([pc for pc in processed_contents if pc.urgency_level == 'high'])}
- Top trending topics: {[topic for topic, count in top_trends]}

Provide JSON response with:
1. overall_summary: Executive summary focusing on business opportunities and technical trends (max 300 words)
2. top_trends: List of top 5 trending topics/technologies
3. high_priority_items: Count of high-urgency items requiring immediate attention

Respond only with valid JSON:"""

        response = self._call_ollama(prompt)
        if response:
            try:
                result = json.loads(response)
                print("   ✅ Executive summary generated successfully")
                return result
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse summary JSON, using fallback")

        # Enhanced fallback with actual data
        print("   ⚠️  Using enhanced fallback summary")
        return {
            'overall_summary': f'Analyzed {len(processed_contents)} authentication-related items with {len([pc for pc in processed_contents if pc.relevance_score > 0.8])} high-relevance discussions. Key focus areas include {", ".join([topic for topic, _ in top_trends[:3]])}.',
            'top_trends': [topic for topic, _ in top_trends],
            'high_priority_items': len([pc for pc in processed_contents if pc.urgency_level == 'high'])
        }
