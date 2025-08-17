"""LLM-driven search strategy engine with GitHub API rate limiting."""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from models import ProcessedContent
from llm_integration import RAGLLMIntegration
from enhanced_purpose_engine import EnhancedRAGPurpose, OpportunityAnalysis
from config import Config


@dataclass
class APIRateLimit:
    """GitHub API rate limiting configuration"""
    max_requests_per_opportunity: int = Config.GITHUB_MAX_REQUESTS_PER_OPPORTUNITY
    max_requests_per_session: int = Config.GITHUB_MAX_REQUESTS_PER_SESSION
    max_requests_per_hour: int = Config.GITHUB_MAX_REQUESTS_PER_HOUR
    requests_used_session: int = 0
    requests_used_current_opportunity: int = 0
    current_opportunity_index: int = 0
    session_start: datetime = None
    last_request_time: datetime = None
    
    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.now()
    
    def start_new_opportunity(self, opportunity_index: int) -> None:
        """Start tracking a new opportunity"""
        self.current_opportunity_index = opportunity_index
        self.requests_used_current_opportunity = 0
    
    def get_remaining_for_opportunity(self) -> int:
        """Get remaining requests for current opportunity"""
        return self.max_requests_per_opportunity - self.requests_used_current_opportunity
    
    def get_remaining_for_session(self) -> int:
        """Get remaining requests for entire session"""
        return self.max_requests_per_session - self.requests_used_session


@dataclass
class SearchContext:
    """Comprehensive context for LLM search decisions"""
    opportunity: ProcessedContent
    analysis: OpportunityAnalysis
    rate_limit: APIRateLimit
    previous_results: List[Dict] = None
    search_history: List[Dict] = None
    success_patterns: List[Dict] = None
    github_capabilities: Dict = None
    
    def to_llm_context(self) -> Dict[str, Any]:
        """Convert to LLM-friendly context"""
        return {
            "opportunity_context": {
                "original_question": f"{self.opportunity.original.title}\n{self.opportunity.original.content[:800]}",
                "extracted_technologies": self.analysis.extracted_technologies,
                "problem_type": self.analysis.problem_type,
                "urgency_level": self.opportunity.urgency_level,
                "business_context": self.analysis.business_context,
                "technical_complexity": self.analysis.technical_complexity,
                "solution_requirements": self.analysis.solution_requirements
            },
            "api_constraints": {
                "requests_remaining_opportunity": self.rate_limit.get_remaining_for_opportunity(),
                "requests_remaining_session": self.rate_limit.get_remaining_for_session(),
                "max_per_opportunity": self.rate_limit.max_requests_per_opportunity,
                "max_per_session": self.rate_limit.max_requests_per_session,
                "requests_used_opportunity": self.rate_limit.requests_used_current_opportunity,
                "requests_used_session": self.rate_limit.requests_used_session,
                "warning": f"Opportunity budget: {self.rate_limit.get_remaining_for_opportunity()}/{self.rate_limit.max_requests_per_opportunity} remaining!"
            },
            "search_capabilities": self.github_capabilities or {
                "available_search_types": ["repositories", "code"],
                "filter_options": ["language", "stars", "updated", "size", "archived"],
                "quality_filters": ["stars:>5", "archived:false", "fork:false"],
                "sort_options": ["stars", "updated", "relevance"],
                "typical_results_per_query": "10-30 repositories",
                "best_practices": [
                    "Use specific terms to reduce noise",
                    "Combine technology + problem type for better results",
                    "Include quality filters to find production-ready code"
                ]
            },
            "learning_context": {
                "previous_search_results": self.previous_results[-3:] if self.previous_results else [],
                "search_patterns": self.search_history[-5:] if self.search_history else [],
                "success_indicators": self.success_patterns or []
            }
        }


@dataclass
class LLMSearchStrategy:
    """LLM-generated search strategy"""
    search_queries: List[Dict[str, Any]]
    reasoning: str
    expected_outcomes: Dict[str, Any]
    success_criteria: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    confidence_score: float
    estimated_api_usage: int


@dataclass
class RepositoryAnalysisStrategy:
    """LLM-generated repository analysis strategy"""
    target_repositories: List[str]
    analysis_depth: str  # shallow, medium, deep
    priority_files: List[str]
    expected_file_patterns: List[str]  # LLM-generated patterns to look for
    analysis_patterns: List[str]
    success_indicators: List[str]
    reasoning: str
    estimated_api_usage: int


class LLMSearchStrategist:
    """LLM-driven search strategy engine with rate limiting"""
    
    def __init__(self, llm_integration: RAGLLMIntegration):
        self.llm_integration = llm_integration
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rate_limit = APIRateLimit()
        self.search_history = []
        self.success_patterns = []
        
    def generate_search_strategy(self, opportunity: ProcessedContent, analysis: OpportunityAnalysis) -> LLMSearchStrategy:
        """Generate comprehensive search strategy using LLM"""
        
        self.logger.info("Generating LLM-driven search strategy")
        
        # Build comprehensive context
        context = SearchContext(
            opportunity=opportunity,
            analysis=analysis,
            rate_limit=self.rate_limit,
            previous_results=getattr(self, 'previous_results', []),
            search_history=self.search_history,
            success_patterns=self.success_patterns
        )
        
        # Create LLM prompt for search strategy
        strategy_prompt = self._create_search_strategy_prompt(context)
        
        # Get LLM decision
        try:
            strategy_data = self.llm_integration.generate_purpose_with_llm(strategy_prompt)
            
            if strategy_data:
                self.logger.info("Successfully generated search strategy via LLM")
                return self._parse_search_strategy(strategy_data, context)
            else:
                self.logger.warning("LLM failed to generate strategy, using fallback")
                return self._fallback_search_strategy(context)
                
        except Exception as e:
            self.logger.error(f"LLM search strategy generation failed: {e}")
            return self._fallback_search_strategy(context)
    
    def generate_repository_analysis_strategy(self, discovered_repos: List[Dict], purpose: EnhancedRAGPurpose) -> RepositoryAnalysisStrategy:
        """Generate LLM-driven repository analysis strategy"""
        
        self.logger.info("Generating LLM-driven repository analysis strategy")
        
        # Build context for repository analysis
        analysis_context = {
            "discovered_repositories": [
                {
                    "name": repo.get("full_name", "unknown"),
                    "stars": repo.get("stargazers_count", 0),
                    "language": repo.get("language", "unknown"),
                    "description": repo.get("description", "")[:200],
                    "updated": repo.get("updated_at", "unknown")
                }
                for repo in discovered_repos[:10]  # Top 10 repos
            ],
            "purpose_context": {
                "primary_purpose": purpose.primary_purpose,
                "technologies": purpose.technologies,
                "complexity": purpose.reasoning.technical_complexity if purpose.reasoning else 5,
                "requirements": purpose.reasoning.solution_requirements if purpose.reasoning else []
            },
            "api_constraints": {
                "requests_remaining": self.rate_limit.get_remaining_for_session(),
                "analysis_budget": min(5, self.rate_limit.get_remaining_for_session() - 2),  # Save 2 for buffer
                "warning": f"Only {self.rate_limit.get_remaining_for_session()} API requests remaining!"
            }
        }
        
        # Create LLM prompt for repository analysis strategy
        analysis_prompt = self._create_repository_analysis_prompt(analysis_context)
        
        try:
            analysis_data = self.llm_integration.generate_purpose_with_llm(analysis_prompt)
            
            if analysis_data:
                self.logger.info("Successfully generated repository analysis strategy via LLM")
                return self._parse_repository_strategy(analysis_data, analysis_context)
            else:
                self.logger.warning("LLM failed to generate repository strategy, using fallback")
                return self._fallback_repository_strategy(analysis_context)
                
        except Exception as e:
            self.logger.error(f"LLM repository strategy generation failed: {e}")
            return self._fallback_repository_strategy(analysis_context)
    
    def track_api_usage(self, requests_used: int) -> bool:
        """Track API usage and check if we can continue"""
        
        self.rate_limit.requests_used_session += requests_used
        self.rate_limit.requests_used_current_opportunity += requests_used
        self.rate_limit.last_request_time = datetime.now()
        
        remaining_opportunity = self.rate_limit.get_remaining_for_opportunity()
        remaining_session = self.rate_limit.get_remaining_for_session()
        
        if remaining_opportunity <= 0:
            self.logger.warning(f"API rate limit reached for current opportunity: {self.rate_limit.requests_used_current_opportunity}/{self.rate_limit.max_requests_per_opportunity}")
            return False
        elif remaining_session <= 0:
            self.logger.warning(f"API rate limit reached for session: {self.rate_limit.requests_used_session}/{self.rate_limit.max_requests_per_session}")
            return False
        elif remaining_opportunity <= 2:
            self.logger.warning(f"Opportunity API limit nearly reached: {remaining_opportunity} requests remaining")
        
        return True
    
    def can_make_request(self, estimated_requests: int = 1) -> bool:
        """Check if we can make the requested number of API calls"""
        
        remaining_opportunity = self.rate_limit.get_remaining_for_opportunity()
        remaining_session = self.rate_limit.get_remaining_for_session()
        
        return remaining_opportunity >= estimated_requests and remaining_session >= estimated_requests
    
    def _create_search_strategy_prompt(self, context: SearchContext) -> str:
        """FIXED: Create search strategy prompt without authentication bias"""
        
        llm_context = context.to_llm_context()
        
        # Determine if this is actually an auth-related query
        problem_type = llm_context["opportunity_context"].get("problem_type", "general")
        technologies = llm_context["opportunity_context"].get("technologies", [])
        is_auth_related = problem_type == "authentication_help" or any(tech in ['jwt', 'oauth', 'auth-general'] for tech in technologies)
        
        problem_description = "authentication challenge" if is_auth_related else f"{problem_type} for {technologies[0] if technologies else 'development'}"
        
        return f"""You are a GitHub search strategist with deep technical expertise. Your goal is to find the most relevant repositories for a developer's {problem_description} using minimal API requests.

OPPORTUNITY ANALYSIS:
{self._format_opportunity_context(llm_context["opportunity_context"])}

IMPORTANT: This is a {problem_type} problem. {"Focus on authentication solutions." if is_auth_related else "Do NOT search for authentication unless explicitly mentioned."}

API CONSTRAINTS (CRITICAL):
- Opportunity Budget: {llm_context["api_constraints"]["requests_remaining_opportunity"]} out of {llm_context["api_constraints"]["max_per_opportunity"]} remaining
- Session Budget: {llm_context["api_constraints"]["requests_remaining_session"]} out of {llm_context["api_constraints"]["max_per_session"]} remaining
- Used This Opportunity: {llm_context["api_constraints"]["requests_used_opportunity"]} requests
- ⚠️  BE STRATEGIC: Each search query costs 1 API request from your {llm_context["api_constraints"]["max_per_opportunity"]}-request opportunity budget!

SEARCH CAPABILITIES:
{self._format_search_capabilities(llm_context["search_capabilities"])}

LEARNING FROM PREVIOUS ATTEMPTS:
{self._format_learning_context(llm_context["learning_context"])}

YOUR TASK:
Generate a strategic search plan for {problem_type} that maximizes results while minimizing API usage.

CRITICAL INSTRUCTIONS:
- If problem_type is "implementation_showcase": Search for similar projects, NOT authentication
- If problem_type is "debugging_issue": Search for debugging/troubleshooting resources
- If problem_type is "authentication_help": Then and only then search for auth solutions
- If problem_type is "performance_optimization": Search for performance examples
- Match your search to the ACTUAL problem type, not authentication by default

RESPOND WITH VALID JSON ONLY:
{{
    "search_queries": [
        {{
            "query": "specific search terms matching {problem_type}",
            "type": "repositories",
            "filters": {{"language": "{technologies[0] if technologies else 'javascript'}", "sort": "stars"}},
            "reasoning": "why this query matches the {problem_type} problem",
            "expected_repos": 8,
            "priority": "high|medium|low"
        }}
    ],
    "reasoning": "Overall strategy for {problem_type} - why these queries will find relevant solutions",
    "expected_outcomes": {{
        "total_repositories": 25,
        "quality_threshold": "production-ready examples",
        "technology_coverage": ["react", "jwt"],
        "solution_completeness": "end-to-end authentication flow"
    }},
    "success_criteria": {{
        "minimum_repos": 5,
        "must_have_features": ["jwt implementation", "error handling"],
        "quality_indicators": ["high stars", "recent updates", "good documentation"]
    }},
    "risk_assessment": {{
        "api_usage_estimate": 3,
        "fallback_plan": "if searches don't yield results",
        "potential_issues": ["too generic queries", "technology mismatch"]
    }},
    "confidence_score": 0.85
}}

STRATEGY GUIDELINES:
1. Use 2-4 strategic queries maximum (save API requests!)
2. Combine technology + problem type for precision
3. Include quality filters to avoid low-quality repos
4. Prioritize queries by expected value
5. Consider the developer's skill level and urgency"""

    def _create_repository_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Create LLM prompt for repository analysis strategy"""
        
        return f"""You are a repository analysis strategist. Given discovered repositories, determine the optimal analysis approach within strict API limits.

DISCOVERED REPOSITORIES:
{self._format_repository_list(context["discovered_repositories"])}

PURPOSE CONTEXT:
- Goal: {context["purpose_context"]["primary_purpose"]}
- Technologies: {', '.join(context["purpose_context"]["technologies"])}
- Complexity Level: {context["purpose_context"]["complexity"]}/10
- Requirements: {', '.join(context["purpose_context"]["requirements"])}

API CONSTRAINTS (CRITICAL):
- Requests Remaining: {context["api_constraints"]["requests_remaining"]}
- Analysis Budget: {context["api_constraints"]["analysis_budget"]} repositories max
- Each repo analysis = 1-3 API requests depending on depth

YOUR TASK:
Select the most promising repositories for detailed analysis and determine the optimal analysis strategy.
Based on the goal and technologies, predict what files and patterns to look for.

RESPOND WITH VALID JSON ONLY:
{{
    "target_repositories": [
        "owner/repo1",
        "owner/repo2"
    ],
    "analysis_depth": "shallow|medium|deep",
    "priority_files": [
        "README.md",
        "auth.js",
        "package.json"
    ],
    "expected_file_patterns": [
        "jwt",
        "auth",
        "token",
        "security",
        "middleware"
    ],
    "analysis_patterns": [
        "jwt implementation patterns",
        "error handling approaches",
        "security best practices"
    ],
    "success_indicators": [
        "find working jwt refresh logic",
        "locate error handling examples",
        "identify security patterns"
    ],
    "reasoning": "Why these specific repositories and analysis approach will provide the best insights",
    "estimated_api_usage": 4
}}

ANALYSIS GUIDELINES:
1. Select 2-4 repositories maximum (API limits!)
2. Prioritize by stars, recency, and technology match
3. Focus on repositories that likely contain complete solutions
4. Balance depth vs breadth based on remaining API calls
5. Look for production-ready code, not just tutorials
6. PREDICT file patterns based on the technology stack and problem type:
   - For React + JWT: look for 'hooks', 'context', 'auth', 'token' files
   - For Spring Security: look for 'config', 'security', 'jwt', 'filter' files
   - For Node.js auth: look for 'middleware', 'auth', 'jwt', 'passport' files
   - For general auth: look for 'login', 'register', 'auth', 'security' files"""

    def _format_opportunity_context(self, context: Dict[str, Any]) -> str:
        """Format opportunity context for LLM"""
        
        return f"""
Original Question: {context["original_question"][:400]}...
Technologies: {', '.join(context["extracted_technologies"])}
Problem Type: {context["problem_type"]}
Technical Complexity: {context["technical_complexity"]}/10
Business Context: {context["business_context"]}
Solution Requirements: {', '.join(context["solution_requirements"][:3])}
Urgency: {context["urgency_level"]}"""

    def _format_search_capabilities(self, capabilities: Dict[str, Any]) -> str:
        """Format search capabilities for LLM"""
        
        return f"""
Available Search Types: {', '.join(capabilities["available_search_types"])}
Quality Filters: {', '.join(capabilities["quality_filters"])}
Sort Options: {', '.join(capabilities["sort_options"])}
Best Practices: {'; '.join(capabilities["best_practices"])}"""

    def _format_learning_context(self, context: Dict[str, Any]) -> str:
        """Format learning context for LLM"""
        
        if not context["previous_search_results"]:
            return "No previous search data available"
        
        return f"""
Recent Search Patterns: {len(context["search_patterns"])} previous searches
Success Indicators: {', '.join([str(p) for p in context["success_indicators"][:3]])}
Previous Results Quality: {len(context["previous_search_results"])} recent results to learn from"""

    def _format_repository_list(self, repos: List[Dict]) -> str:
        """Format repository list for LLM"""
        
        formatted = []
        for i, repo in enumerate(repos[:8], 1):  # Show top 8
            formatted.append(
                f"{i}. {repo['name']} (⭐{repo['stars']} | {repo['language']} | {repo['description'][:60]}...)"
            )
        
        return '\n'.join(formatted)

    def _parse_search_strategy(self, strategy_data: Dict[str, Any], context: SearchContext) -> LLMSearchStrategy:
        """Parse LLM response into search strategy"""
        
        return LLMSearchStrategy(
            search_queries=strategy_data.get('search_queries', []),
            reasoning=strategy_data.get('reasoning', 'LLM-generated search strategy'),
            expected_outcomes=strategy_data.get('expected_outcomes', {}),
            success_criteria=strategy_data.get('success_criteria', {}),
            risk_assessment=strategy_data.get('risk_assessment', {}),
            confidence_score=strategy_data.get('confidence_score', 0.7),
            estimated_api_usage=strategy_data.get('risk_assessment', {}).get('api_usage_estimate', 3)
        )

    def _parse_repository_strategy(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> RepositoryAnalysisStrategy:
        """Parse LLM response into repository analysis strategy"""
        
        return RepositoryAnalysisStrategy(
            target_repositories=analysis_data.get('target_repositories', []),
            analysis_depth=analysis_data.get('analysis_depth', 'medium'),
            priority_files=analysis_data.get('priority_files', ['README.md']),
            expected_file_patterns=analysis_data.get('expected_file_patterns', ['auth', 'login', 'token']),
            analysis_patterns=analysis_data.get('analysis_patterns', []),
            success_indicators=analysis_data.get('success_indicators', []),
            reasoning=analysis_data.get('reasoning', 'LLM-generated analysis strategy'),
            estimated_api_usage=analysis_data.get('estimated_api_usage', 3)
        )

    def _fallback_search_strategy(self, context: SearchContext) -> LLMSearchStrategy:
        """Fallback search strategy if LLM fails"""
        
        primary_tech = context.analysis.extracted_technologies[0] if context.analysis.extracted_technologies else 'javascript'
        
        return LLMSearchStrategy(
            search_queries=[
                {
                    "query": f"{primary_tech} authentication",
                    "type": "repositories",
                    "filters": {"language": primary_tech, "sort": "stars"},
                    "reasoning": "Fallback: primary technology authentication",
                    "expected_repos": 10,
                    "priority": "high"
                }
            ],
            reasoning="Fallback strategy due to LLM failure",
            expected_outcomes={"total_repositories": 10},
            success_criteria={"minimum_repos": 3},
            risk_assessment={"api_usage_estimate": 1},
            confidence_score=0.5,
            estimated_api_usage=1
        )

    def _fallback_repository_strategy(self, context: Dict[str, Any]) -> RepositoryAnalysisStrategy:
        """Fallback repository analysis strategy if LLM fails"""
        
        # Pick top 2 repositories by stars
        top_repos = sorted(context["discovered_repositories"], 
                          key=lambda r: r.get("stars", 0), reverse=True)[:2]
        
        # Determine fallback file patterns based on technologies
        technologies = context["purpose_context"]["technologies"]
        fallback_patterns = ['auth', 'login', 'token', 'security']
        
        if any(tech in ['react', 'vue', 'angular'] for tech in technologies):
            fallback_patterns.extend(['hooks', 'context', 'component'])
        if any(tech in ['spring', 'java'] for tech in technologies):
            fallback_patterns.extend(['config', 'filter', 'controller'])
        if any(tech in ['node', 'express'] for tech in technologies):
            fallback_patterns.extend(['middleware', 'route', 'passport'])
        
        return RepositoryAnalysisStrategy(
            target_repositories=[repo["name"] for repo in top_repos],
            analysis_depth="shallow",
            priority_files=["README.md", "package.json"],
            expected_file_patterns=fallback_patterns[:8],  # Top 8 patterns
            analysis_patterns=["basic authentication patterns"],
            success_indicators=["find working examples"],
            reasoning="Fallback strategy due to LLM failure",
            estimated_api_usage=2
        )

    def update_search_history(self, strategy: LLMSearchStrategy, results: List[Dict]) -> None:
        """Update search history for learning"""
        
        search_record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": asdict(strategy),
            "results_count": len(results),
            "api_requests_used": strategy.estimated_api_usage,
            "success_score": self._calculate_success_score(strategy, results)
        }
        
        self.search_history.append(search_record)
        
        # Keep only recent history
        if len(self.search_history) > 20:
            self.search_history = self.search_history[-20:]

    def _calculate_success_score(self, strategy: LLMSearchStrategy, results: List[Dict]) -> float:
        """Calculate success score for learning"""
        
        expected = strategy.expected_outcomes.get('total_repositories', 10)
        actual = len(results)
        
        if expected == 0:
            return 0.5
        
        ratio_score = min(actual / expected, 1.0)
        
        # Bonus for high-quality results (high stars)
        quality_bonus = 0.1 if any(repo.get('stargazers_count', 0) > 100 for repo in results) else 0
        
        return min(ratio_score + quality_bonus, 1.0)
