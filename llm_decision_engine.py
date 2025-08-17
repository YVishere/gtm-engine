"""LLM-driven decision engine for RAG email system quality improvements."""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from llm_integration import RAGLLMIntegration


class LLMDecisionEngine:
    """Core LLM decision engine for technology detection, problem classification, and email relevance."""
    
    def __init__(self, llm_integration: RAGLLMIntegration):
        self.llm_integration = llm_integration
        self.logger = logging.getLogger(self.__class__.__name__)
        self.decision_cache = {}  # Simple caching for performance
        
    def llm_technology_detection(self, content: str) -> Dict[str, Any]:
        """Use LLM to detect technologies mentioned in content with high accuracy."""
        
        # Simple cache check
        cache_key = f"tech_{hash(content[:500])}"
        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]
        
        prompt = f"""
Analyze this developer question and identify ONLY the technologies explicitly mentioned or clearly implied:

CONTENT: {content}

Rules:
- Only extract technologies actually discussed in the content
- Look for: code snippets, package names, framework mentions, tool names
- Do NOT assume technologies based on problem type
- Do NOT add authentication technologies unless explicitly mentioned
- Be conservative - prefer fewer accurate technologies over many guessed ones

Return JSON:
{{
    "primary_technologies": ["tech1", "tech2"],
    "confidence_scores": {{"tech1": 0.95, "tech2": 0.80}},
    "reasoning": "Why these specific technologies were identified",
    "authentication_mentioned": true/false
}}
"""
        
        try:
            result = self.llm_integration.generate_purpose_with_llm(prompt)
            if result and self._validate_tech_response(result):
                self.decision_cache[cache_key] = result
                self.logger.info(f"LLM detected technologies: {result.get('primary_technologies', [])}")
                return result
            else:
                self.logger.warning("LLM tech detection failed validation, using fallback")
                return self._fallback_tech_detection(content)
        except Exception as e:
            self.logger.error(f"LLM tech detection failed: {e}")
            return self._fallback_tech_detection(content)
    
    def llm_problem_classification(self, content: str) -> Dict[str, Any]:
        """Use LLM to classify the actual problem type based on user intent."""
        
        prompt = f"""
Classify this developer question based on their PRIMARY intent:

CONTENT: {content}

CATEGORIES:
- authentication_help: User needs help implementing login/auth systems
- debugging_issue: Technical error/bug that needs fixing
- implementation_showcase: Sharing/demoing their work  
- architecture_decision: Asking for best practices/technology advice
- performance_optimization: Speed/efficiency improvements
- integration_challenge: Connecting different systems/technologies
- general_consultation: Broad technical discussion

Focus on what the user PRIMARILY wants to achieve, not secondary mentions.

Return JSON:
{{
    "problem_type": "category",
    "confidence": 0.90,
    "reasoning": "Why this classification was chosen",
    "secondary_intents": ["other possible intents"],
    "business_context": "personal/startup/enterprise/academic"
}}
"""
        
        try:
            result = self.llm_integration.generate_purpose_with_llm(prompt)
            if result and self._validate_classification_response(result):
                self.logger.info(f"LLM classified problem as: {result.get('problem_type')}")
                return result
            else:
                self.logger.warning("LLM classification failed validation, using fallback")
                return self._fallback_classification(content)
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            return self._fallback_classification(content)
    
    def llm_email_relevance_assessment(self, query: str, technologies: List[str], problem_type: str) -> Dict:
        """Use LLM to determine email relevance and business value."""
        
        prompt = f"""
Determine if an authentication solution consultation email would be RELEVANT and VALUABLE:

ORIGINAL QUERY: {query}
DETECTED TECHNOLOGIES: {technologies}
PROBLEM TYPE: {problem_type}

Evaluation Criteria:
1. Does the user actually need authentication/security help? (not just mention it)
2. Are they implementing/evaluating auth solutions?
3. Would technical authentication consultation provide genuine value?
4. Is this a potential business opportunity (not academic/theoretical)?
5. Does the query indicate decision-making authority or implementation responsibility?

Return JSON:
{{
    "should_send_email": true/false,
    "relevance_score": 0.85,
    "business_value": "high/medium/low",
    "reasoning": "Detailed explanation of decision",
    "confidence": 0.90,
    "auth_signals": ["specific indicators that suggest auth need"],
    "non_auth_signals": ["indicators this isn't about auth"]
}}
"""
        
        try:
            result = self.llm_integration.generate_purpose_with_llm(prompt)
            if result and self._validate_relevance_response(result):
                self.logger.info(f"LLM relevance assessment: {result.get('relevance_score', 0):.2f}")
                return result
            else:
                self.logger.warning("LLM relevance assessment failed validation, using fallback")
                return self._fallback_relevance_scoring(query, problem_type)
        except Exception as e:
            self.logger.error(f"LLM relevance assessment failed: {e}")
            return self._fallback_relevance_scoring(query, problem_type)
    
    def safe_llm_decision(self, llm_function, fallback_function, *args):
        """Safe LLM decision with automatic fallback."""
        try:
            result = llm_function(*args)
            if self._validate_llm_response(result):
                return result
            else:
                self.logger.warning("LLM response validation failed, using fallback")
                return fallback_function(*args)
        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}, using fallback")
            return fallback_function(*args)
    
    def _validate_tech_response(self, response: Dict) -> bool:
        """Validate technology detection response."""
        if not isinstance(response, dict):
            return False
        
        required_fields = ['primary_technologies', 'authentication_mentioned']
        for field in required_fields:
            if field not in response:
                return False
        
        if not isinstance(response['primary_technologies'], list):
            return False
            
        return True
    
    def _validate_classification_response(self, response: Dict) -> bool:
        """Validate problem classification response."""
        if not isinstance(response, dict):
            return False
        
        valid_categories = [
            'authentication_help', 'debugging_issue', 'implementation_showcase',
            'architecture_decision', 'performance_optimization', 'integration_challenge',
            'general_consultation'
        ]
        
        return (response.get('problem_type') in valid_categories and
                isinstance(response.get('confidence', 0), (int, float)))
    
    def _validate_relevance_response(self, response: Dict) -> bool:
        """Validate email relevance response."""
        if not isinstance(response, dict):
            return False
        
        required_fields = ['should_send_email', 'relevance_score']
        for field in required_fields:
            if field not in response:
                return False
        
        return (isinstance(response['should_send_email'], bool) and
                isinstance(response['relevance_score'], (int, float)))
    
    def _validate_llm_response(self, response: Any) -> bool:
        """General LLM response validation."""
        if not isinstance(response, dict):
            return False
        
        # Check for common response patterns
        if 'technologies' in response:
            return self._validate_tech_response(response)
        elif 'problem_type' in response:
            return self._validate_classification_response(response)
        elif 'should_send_email' in response:
            return self._validate_relevance_response(response)
        
        return True
    
    def _fallback_tech_detection(self, content: str) -> Dict[str, Any]:
        """Fallback technology detection using pattern matching."""
        content_lower = content.lower()
        
        # Basic pattern matching
        detected_tech = []
        if 'typo3' in content_lower:
            detected_tech.extend(['php', 'typo3'])
        elif 'webassembly' in content_lower or 'wasm' in content_lower:
            detected_tech.extend(['webassembly', 'c++'])
        elif 'react' in content_lower:
            detected_tech.append('react')
        elif 'angular' in content_lower:
            detected_tech.append('angular')
        elif 'vue' in content_lower:
            detected_tech.append('vue')
        else:
            detected_tech.append('javascript')  # Default fallback
        
        return {
            'primary_technologies': detected_tech,
            'confidence_scores': {tech: 0.7 for tech in detected_tech},
            'reasoning': 'Fallback pattern matching',
            'authentication_mentioned': any(auth in content_lower 
                                           for auth in ['auth', 'login', 'jwt', 'oauth'])
        }
    
    def _fallback_classification(self, content: str) -> Dict[str, Any]:
        """Fallback problem classification."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['error', 'bug', 'issue', 'problem']):
            problem_type = 'debugging_issue'
        elif any(word in content_lower for word in ['auth', 'login', 'jwt', 'oauth']):
            problem_type = 'authentication_help'
        elif any(word in content_lower for word in ['showcase', 'demo', 'project']):
            problem_type = 'implementation_showcase'
        else:
            problem_type = 'general_consultation'
        
        return {
            'problem_type': problem_type,
            'confidence': 0.6,
            'reasoning': 'Fallback keyword matching',
            'secondary_intents': [],
            'business_context': 'unknown'
        }
    
    def _fallback_relevance_scoring(self, query: str, problem_type: str) -> Dict:
        """Fallback email relevance scoring."""
        query_lower = query.lower()
        
        # Simple heuristic scoring
        auth_keywords = ['authentication', 'login', 'jwt', 'oauth', 'auth', 'security']
        auth_score = sum(1 for keyword in auth_keywords if keyword in query_lower)
        
        base_score = 0.4 if problem_type == 'authentication_help' else 0.2
        relevance_score = min(base_score + (auth_score * 0.1), 1.0)
        
        should_send = (problem_type == 'authentication_help' and auth_score >= 2) or relevance_score >= 0.7
        
        return {
            'should_send_email': should_send,
            'relevance_score': relevance_score,
            'business_value': 'medium' if should_send else 'low',
            'reasoning': f'Fallback scoring based on {auth_score} auth keywords',
            'confidence': 0.6,
            'auth_signals': [f'{auth_score} auth keywords found'],
            'non_auth_signals': []
        }
    
    def track_llm_performance(self, decision_type: str, llm_result: Any, fallback_used: bool = False):
        """Track LLM decision performance for analytics."""
        performance_data = {
            'decision_type': decision_type,
            'llm_used': not fallback_used,
            'timestamp': datetime.now().isoformat(),
            'success': llm_result is not None
        }
        
        self.logger.info(f"LLM Performance: {decision_type} - {'Success' if not fallback_used else 'Fallback'}")
        return performance_data
