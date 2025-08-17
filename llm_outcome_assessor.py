"""LLM-driven outcome assessment and adaptive learning system."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from llm_integration import RAGLLMIntegration
from llm_search_strategist import LLMSearchStrategy, RepositoryAnalysisStrategy
from enhanced_action_transparency import SearchExecutionResult, RepositoryAnalysisResult


@dataclass
class OutcomeAssessment:
    """Comprehensive LLM-generated outcome assessment"""
    overall_success_score: float
    strengths: List[str]
    weaknesses: List[str]
    gap_analysis: str
    recommendations: List[str]
    lessons_learned: List[str]
    strategy_effectiveness: Dict[str, float]
    next_actions: List[str]
    confidence: float
    timestamp: datetime


@dataclass
class AdaptiveLearning:
    """Learning patterns from LLM assessments"""
    successful_patterns: List[Dict[str, Any]]
    failure_patterns: List[Dict[str, Any]]
    technology_insights: Dict[str, Any]
    query_effectiveness: Dict[str, float]
    repository_quality_indicators: List[str]


class LLMOutcomeAssessor:
    """LLM-driven outcome assessment and learning system"""
    
    def __init__(self, llm_integration: RAGLLMIntegration):
        self.llm_integration = llm_integration
        self.logger = logging.getLogger(self.__class__.__name__)
        self.learning_database = AdaptiveLearning(
            successful_patterns=[],
            failure_patterns=[],
            technology_insights={},
            query_effectiveness={},
            repository_quality_indicators=[]
        )
    
    def assess_search_outcome(self, 
                             strategy: LLMSearchStrategy,
                             execution_result: SearchExecutionResult,
                             original_purpose: str) -> OutcomeAssessment:
        """Comprehensive LLM assessment of search outcome"""
        
        self.logger.info("Generating LLM-driven search outcome assessment")
        
        # Build comprehensive context for LLM assessment
        assessment_context = self._build_search_assessment_context(
            strategy, execution_result, original_purpose
        )
        
        # Create LLM prompt for outcome assessment
        assessment_prompt = self._create_search_assessment_prompt(assessment_context)
        
        try:
            assessment_data = self.llm_integration.generate_purpose_with_llm(assessment_prompt)
            
            if assessment_data:
                self.logger.info("Successfully generated search assessment via LLM")
                outcome = self._parse_search_assessment(assessment_data)
                
                # Learn from this outcome
                self._update_learning_patterns(strategy, execution_result, outcome)
                
                return outcome
            else:
                self.logger.warning("LLM failed to generate assessment, using fallback")
                return self._fallback_search_assessment(strategy, execution_result)
                
        except Exception as e:
            self.logger.error(f"LLM search assessment failed: {e}")
            return self._fallback_search_assessment(strategy, execution_result)
    
    def assess_repository_analysis_outcome(self,
                                         strategy: RepositoryAnalysisStrategy,
                                         analysis_result: RepositoryAnalysisResult,
                                         original_purpose: str) -> OutcomeAssessment:
        """Comprehensive LLM assessment of repository analysis outcome"""
        
        self.logger.info("Generating LLM-driven repository analysis assessment")
        
        # Build context for repository analysis assessment
        assessment_context = self._build_repository_assessment_context(
            strategy, analysis_result, original_purpose
        )
        
        # Create LLM prompt for repository assessment
        assessment_prompt = self._create_repository_assessment_prompt(assessment_context)
        
        try:
            assessment_data = self.llm_integration.generate_purpose_with_llm(assessment_prompt)
            
            if assessment_data:
                self.logger.info("Successfully generated repository assessment via LLM")
                outcome = self._parse_repository_assessment(assessment_data)
                
                # Learn from this outcome
                self._update_repository_learning(strategy, analysis_result, outcome)
                
                return outcome
            else:
                self.logger.warning("LLM failed to generate repository assessment, using fallback")
                return self._fallback_repository_assessment(strategy, analysis_result)
                
        except Exception as e:
            self.logger.error(f"LLM repository assessment failed: {e}")
            return self._fallback_repository_assessment(strategy, analysis_result)
    
    def generate_adaptive_recommendations(self, 
                                        current_results: List[Dict],
                                        remaining_api_calls: int,
                                        original_goal: str) -> Dict[str, Any]:
        """Generate LLM-driven adaptive recommendations for next steps"""
        
        self.logger.info("Generating adaptive recommendations via LLM")
        
        # Build context with learning insights
        recommendation_context = {
            "current_results": {
                "repositories_found": len(current_results),
                "quality_indicators": self._assess_result_quality(current_results),
                "technology_coverage": self._assess_technology_coverage(current_results),
                "completeness_score": self._assess_completeness(current_results, original_goal)
            },
            "constraints": {
                "remaining_api_calls": remaining_api_calls,
                "time_pressure": "medium",  # Could be dynamic
                "quality_threshold": 0.7
            },
            "learning_insights": {
                "successful_patterns": self.learning_database.successful_patterns[-3:],
                "query_effectiveness": dict(list(self.learning_database.query_effectiveness.items())[-5:]),
                "quality_indicators": self.learning_database.repository_quality_indicators[-5:]
            },
            "original_goal": original_goal
        }
        
        # Create recommendation prompt
        recommendation_prompt = self._create_recommendation_prompt(recommendation_context)
        
        try:
            recommendations = self.llm_integration.generate_purpose_with_llm(recommendation_prompt)
            
            if recommendations:
                self.logger.info("Successfully generated adaptive recommendations via LLM")
                return recommendations
            else:
                return self._fallback_recommendations(recommendation_context)
                
        except Exception as e:
            self.logger.error(f"LLM recommendation generation failed: {e}")
            return self._fallback_recommendations(recommendation_context)
    
    def _build_search_assessment_context(self, 
                                       strategy: LLMSearchStrategy,
                                       execution_result: SearchExecutionResult,
                                       original_purpose: str) -> Dict[str, Any]:
        """Build comprehensive context for search assessment"""
        
        return {
            "original_purpose": original_purpose,
            "strategy_planned": {
                "queries": [q.get('query', '') for q in strategy.search_queries],
                "expected_repos": strategy.expected_outcomes.get('total_repositories', 0),
                "confidence": strategy.confidence_score,
                "api_budget": strategy.estimated_api_usage,
                "reasoning": strategy.reasoning
            },
            "execution_results": {
                "queries_executed": len(execution_result.queries_executed),
                "repositories_found": len(execution_result.repositories_found),
                "api_requests_used": execution_result.api_requests_used,
                "execution_time": execution_result.execution_time,
                "top_repositories": [
                    {
                        "name": repo.get('full_name', 'unknown'),
                        "stars": repo.get('stargazers_count', 0),
                        "language": repo.get('language', 'unknown'),
                        "description": repo.get('description', '')[:100]
                    }
                    for repo in sorted(execution_result.repositories_found, 
                                     key=lambda r: r.get('stargazers_count', 0), 
                                     reverse=True)[:5]
                ]
            },
            "success_metrics": execution_result.success_metrics,
            "gaps_identified": execution_result.gaps_identified
        }
    
    def _build_repository_assessment_context(self,
                                           strategy: RepositoryAnalysisStrategy,
                                           analysis_result: RepositoryAnalysisResult,
                                           original_purpose: str) -> Dict[str, Any]:
        """Build context for repository analysis assessment"""
        
        return {
            "original_purpose": original_purpose,
            "strategy_planned": {
                "target_repositories": strategy.target_repositories,
                "analysis_depth": strategy.analysis_depth,
                "priority_files": strategy.priority_files,
                "expected_patterns": strategy.analysis_patterns,
                "reasoning": strategy.reasoning,
                "api_budget": strategy.estimated_api_usage
            },
            "analysis_results": {
                "repositories_analyzed": analysis_result.repositories_analyzed,
                "files_examined": analysis_result.files_examined,
                "patterns_found": analysis_result.code_patterns_found,
                "insights_extracted": analysis_result.insights_extracted,
                "quality_score": analysis_result.analysis_quality,
                "api_requests_used": analysis_result.api_requests_used
            }
        }
    
    def _create_search_assessment_prompt(self, context: Dict[str, Any]) -> str:
        """Create LLM prompt for search outcome assessment"""
        
        return f"""You are an expert technical search analyst. Assess the effectiveness of a GitHub search strategy and its execution results.

ORIGINAL PURPOSE:
{context['original_purpose']}

STRATEGY PLANNED:
- Queries: {', '.join(context['strategy_planned']['queries'])}
- Expected Repositories: {context['strategy_planned']['expected_repos']}
- API Budget: {context['strategy_planned']['api_budget']} requests
- Confidence: {context['strategy_planned']['confidence']:.2f}
- Reasoning: {context['strategy_planned']['reasoning']}

EXECUTION RESULTS:
- Queries Executed: {context['execution_results']['queries_executed']}
- Repositories Found: {context['execution_results']['repositories_found']}
- API Requests Used: {context['execution_results']['api_requests_used']}
- Execution Time: {context['execution_results']['execution_time']:.2f}s

TOP REPOSITORIES FOUND:
{self._format_repositories_for_prompt(context['execution_results']['top_repositories'])}

SUCCESS METRICS:
{context['success_metrics']}

GAPS IDENTIFIED:
{'; '.join(context['gaps_identified']) if context['gaps_identified'] else 'None identified'}

YOUR TASK:
Provide a comprehensive assessment of this search strategy and execution. Consider:
1. Did the strategy align with the original purpose?
2. Were the results quality and quantity appropriate?
3. Was API usage efficient?
4. What gaps exist and how can they be addressed?

RESPOND WITH VALID JSON ONLY:
{{
    "overall_success_score": 0.85,
    "strengths": [
        "Found high-quality repositories with good star ratings",
        "Efficient API usage within budget"
    ],
    "weaknesses": [
        "Limited technology diversity in results",
        "Missing specific implementation patterns"
    ],
    "gap_analysis": "Detailed analysis of what's missing and why",
    "recommendations": [
        "Try more specific queries targeting [specific need]",
        "Include additional filters for [specific criteria]"
    ],
    "lessons_learned": [
        "Technology X queries work better with Y filters",
        "Star count correlation with code quality is strong"
    ],
    "strategy_effectiveness": {{
        "query_quality": 0.8,
        "result_relevance": 0.9,
        "api_efficiency": 0.7,
        "goal_alignment": 0.85
    }},
    "next_actions": [
        "Search for more specific authentication patterns",
        "Analyze top 3 repositories in detail"
    ],
    "confidence": 0.9
}}

ASSESSMENT GUIDELINES:
- overall_success_score: 0.0-1.0 based on goal achievement
- Be specific about what worked and what didn't
- Provide actionable recommendations
- Consider API efficiency in your assessment
- Focus on practical next steps"""

    def _create_repository_assessment_prompt(self, context: Dict[str, Any]) -> str:
        """Create LLM prompt for repository analysis assessment"""
        
        return f"""You are an expert code analysis reviewer. Assess the effectiveness of a repository analysis strategy and results.

ORIGINAL PURPOSE:
{context['original_purpose']}

ANALYSIS STRATEGY:
- Target Repositories: {', '.join(context['strategy_planned']['target_repositories'])}
- Analysis Depth: {context['strategy_planned']['analysis_depth']}
- Priority Files: {', '.join(context['strategy_planned']['priority_files'])}
- Expected Patterns: {', '.join(context['strategy_planned']['expected_patterns'])}
- API Budget: {context['strategy_planned']['api_budget']} requests

ANALYSIS RESULTS:
- Repositories Analyzed: {len(context['analysis_results']['repositories_analyzed'])}
- Files Examined: {len(context['analysis_results']['files_examined'])}
- Code Patterns Found: {len(context['analysis_results']['patterns_found'])}
- Insights Extracted: {len(context['analysis_results']['insights_extracted'])}
- Quality Score: {context['analysis_results']['quality_score']:.2f}
- API Requests Used: {context['analysis_results']['api_requests_used']}

KEY FINDINGS:
Patterns: {'; '.join(context['analysis_results']['patterns_found'][:3])}
Insights: {'; '.join(context['analysis_results']['insights_extracted'][:3])}

YOUR TASK:
Assess how well the repository analysis met the original purpose and provide guidance for improvement.

RESPOND WITH VALID JSON ONLY:
{{
    "overall_success_score": 0.8,
    "strengths": [
        "Found relevant authentication patterns",
        "Covered multiple implementation approaches"
    ],
    "weaknesses": [
        "Limited depth in error handling analysis",
        "Missing production-ready examples"
    ],
    "gap_analysis": "Analysis of what information is still needed",
    "recommendations": [
        "Dive deeper into error handling patterns",
        "Look for production deployment examples"
    ],
    "lessons_learned": [
        "Popular repositories often have better documentation",
        "TypeScript repos provide better type safety examples"
    ],
    "strategy_effectiveness": {{
        "repository_selection": 0.9,
        "analysis_depth": 0.7,
        "pattern_identification": 0.8,
        "insight_extraction": 0.75
    }},
    "next_actions": [
        "Analyze additional files in top repository",
        "Search for specific error handling patterns"
    ],
    "confidence": 0.85
}}"""

    def _create_recommendation_prompt(self, context: Dict[str, Any]) -> str:
        """Create LLM prompt for adaptive recommendations"""
        
        return f"""You are an adaptive search optimization expert. Given current results and constraints, recommend the optimal next steps.

CURRENT SITUATION:
- Repositories Found: {context['current_results']['repositories_found']}
- Quality Score: {context['current_results']['quality_indicators']}
- Technology Coverage: {context['current_results']['technology_coverage']}
- Completeness: {context['current_results']['completeness_score']:.2f}

CONSTRAINTS:
- Remaining API Calls: {context['constraints']['remaining_api_calls']}
- Quality Threshold: {context['constraints']['quality_threshold']}

LEARNING INSIGHTS:
Recent Successful Patterns: {len(context['learning_insights']['successful_patterns'])} patterns
Query Effectiveness: {context['learning_insights']['query_effectiveness']}

ORIGINAL GOAL:
{context['original_goal']}

YOUR TASK:
Recommend the most effective next steps given the constraints and learning insights.

RESPOND WITH VALID JSON ONLY:
{{
    "should_continue": true,
    "recommended_actions": [
        {{
            "action_type": "search|analyze|stop",
            "description": "Specific action to take",
            "reasoning": "Why this action is optimal",
            "api_cost": 2,
            "expected_value": 0.8
        }}
    ],
    "priority_focus": "What to focus on next",
    "risk_assessment": "Potential risks and mitigation",
    "success_probability": 0.75,
    "alternative_strategies": [
        "Backup plan if main recommendation fails"
    ]
}}"""

    def _format_repositories_for_prompt(self, repositories: List[Dict]) -> str:
        """Format repositories for LLM prompt"""
        
        if not repositories:
            return "No repositories found"
        
        formatted = []
        for i, repo in enumerate(repositories, 1):
            formatted.append(f"{i}. {repo['name']} (⭐{repo['stars']} | {repo['language']} | {repo['description'][:60]}...)")
        
        return '\n'.join(formatted)

    def _parse_search_assessment(self, assessment_data: Dict[str, Any]) -> OutcomeAssessment:
        """Parse LLM search assessment response"""
        
        return OutcomeAssessment(
            overall_success_score=assessment_data.get('overall_success_score', 0.5),
            strengths=assessment_data.get('strengths', []),
            weaknesses=assessment_data.get('weaknesses', []),
            gap_analysis=assessment_data.get('gap_analysis', ''),
            recommendations=assessment_data.get('recommendations', []),
            lessons_learned=assessment_data.get('lessons_learned', []),
            strategy_effectiveness=assessment_data.get('strategy_effectiveness', {}),
            next_actions=assessment_data.get('next_actions', []),
            confidence=assessment_data.get('confidence', 0.5),
            timestamp=datetime.now()
        )

    def _parse_repository_assessment(self, assessment_data: Dict[str, Any]) -> OutcomeAssessment:
        """Parse LLM repository assessment response"""
        
        return OutcomeAssessment(
            overall_success_score=assessment_data.get('overall_success_score', 0.5),
            strengths=assessment_data.get('strengths', []),
            weaknesses=assessment_data.get('weaknesses', []),
            gap_analysis=assessment_data.get('gap_analysis', ''),
            recommendations=assessment_data.get('recommendations', []),
            lessons_learned=assessment_data.get('lessons_learned', []),
            strategy_effectiveness=assessment_data.get('strategy_effectiveness', {}),
            next_actions=assessment_data.get('next_actions', []),
            confidence=assessment_data.get('confidence', 0.5),
            timestamp=datetime.now()
        )

    def _fallback_search_assessment(self, strategy: LLMSearchStrategy, execution_result: SearchExecutionResult) -> OutcomeAssessment:
        """Fallback assessment if LLM fails"""
        
        repos_found = len(execution_result.repositories_found)
        expected = strategy.expected_outcomes.get('total_repositories', 10)
        success_score = min(repos_found / max(expected, 1), 1.0)
        
        return OutcomeAssessment(
            overall_success_score=success_score,
            strengths=["Search completed successfully"] if repos_found > 0 else [],
            weaknesses=["Limited assessment available"] if repos_found < expected else [],
            gap_analysis="Fallback assessment - limited analysis available",
            recommendations=["Review results manually"],
            lessons_learned=[],
            strategy_effectiveness={"overall": success_score},
            next_actions=["Continue with repository analysis"],
            confidence=0.3,
            timestamp=datetime.now()
        )

    def _fallback_repository_assessment(self, strategy: RepositoryAnalysisStrategy, analysis_result: RepositoryAnalysisResult) -> OutcomeAssessment:
        """Fallback repository assessment if LLM fails"""
        
        return OutcomeAssessment(
            overall_success_score=analysis_result.analysis_quality,
            strengths=["Repository analysis completed"],
            weaknesses=["Limited assessment available"],
            gap_analysis="Fallback assessment - limited analysis available",
            recommendations=["Review analysis results manually"],
            lessons_learned=[],
            strategy_effectiveness={"overall": analysis_result.analysis_quality},
            next_actions=["Proceed with email generation"],
            confidence=0.3,
            timestamp=datetime.now()
        )

    def _fallback_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback recommendations if LLM fails"""
        
        remaining_calls = context['constraints']['remaining_api_calls']
        
        return {
            "should_continue": remaining_calls > 2,
            "recommended_actions": [
                {
                    "action_type": "analyze" if remaining_calls > 3 else "stop",
                    "description": "Analyze top repositories" if remaining_calls > 3 else "Proceed with current results",
                    "reasoning": "Fallback recommendation",
                    "api_cost": min(remaining_calls - 1, 3),
                    "expected_value": 0.6
                }
            ],
            "priority_focus": "Make best use of remaining API calls",
            "risk_assessment": "Limited assessment due to LLM failure",
            "success_probability": 0.5,
            "alternative_strategies": ["Manual review of current results"]
        }

    def _update_learning_patterns(self, strategy: LLMSearchStrategy, execution_result: SearchExecutionResult, assessment: OutcomeAssessment) -> None:
        """Update learning database with search patterns"""
        
        pattern = {
            "timestamp": datetime.now().isoformat(),
            "strategy": {
                "queries": [q.get('query', '') for q in strategy.search_queries],
                "technologies": strategy.expected_outcomes.get('technology_coverage', []),
                "approach": strategy.reasoning
            },
            "results": {
                "repositories_found": len(execution_result.repositories_found),
                "api_efficiency": len(execution_result.repositories_found) / max(execution_result.api_requests_used, 1),
                "execution_time": execution_result.execution_time
            },
            "success_score": assessment.overall_success_score
        }
        
        if assessment.overall_success_score >= 0.7:
            self.learning_database.successful_patterns.append(pattern)
            # Keep only recent successful patterns
            if len(self.learning_database.successful_patterns) > 10:
                self.learning_database.successful_patterns = self.learning_database.successful_patterns[-10:]
        else:
            self.learning_database.failure_patterns.append(pattern)
            # Keep only recent failure patterns
            if len(self.learning_database.failure_patterns) > 5:
                self.learning_database.failure_patterns = self.learning_database.failure_patterns[-5:]
        
        # Update query effectiveness
        for query_info in strategy.search_queries:
            query = query_info.get('query', '')
            if query:
                self.learning_database.query_effectiveness[query] = assessment.overall_success_score

    def _update_repository_learning(self, strategy: RepositoryAnalysisStrategy, analysis_result: RepositoryAnalysisResult, assessment: OutcomeAssessment) -> None:
        """Update learning database with repository analysis patterns"""
        
        # Update quality indicators based on successful analyses
        if assessment.overall_success_score >= 0.7:
            for repo in analysis_result.repositories_analyzed:
                self.learning_database.repository_quality_indicators.append(f"Successful analysis: {repo}")
            
            # Keep only recent indicators
            if len(self.learning_database.repository_quality_indicators) > 15:
                self.learning_database.repository_quality_indicators = self.learning_database.repository_quality_indicators[-15:]

    def _assess_result_quality(self, results: List[Dict]) -> float:
        """Assess overall quality of search results"""
        
        if not results:
            return 0.0
        
        # Simple quality assessment based on stars and recency
        total_score = 0
        for repo in results:
            stars = repo.get('stargazers_count', 0)
            star_score = min(stars / 1000, 1.0)  # Normalize to 1.0 for 1000+ stars
            
            # Add language relevance (simplified)
            lang_score = 0.1 if repo.get('language') else 0
            
            total_score += star_score + lang_score
        
        return min(total_score / len(results), 1.0)

    def _assess_technology_coverage(self, results: List[Dict]) -> float:
        """Assess technology coverage in results"""
        
        if not results:
            return 0.0
        
        languages = set()
        for repo in results:
            lang = repo.get('language')
            if lang:
                languages.add(lang.lower())
        
        # Simple coverage assessment
        return min(len(languages) / 3, 1.0)  # Normalize for 3+ languages

    def _assess_completeness(self, results: List[Dict], original_goal: str) -> float:
        """Assess how well results match the original goal"""
        
        if not results or not original_goal:
            return 0.5
        
        # Simple keyword matching for now
        goal_words = set(original_goal.lower().split())
        
        match_scores = []
        for repo in results:
            repo_text = f"{repo.get('name', '')} {repo.get('description', '')}".lower()
            repo_words = set(repo_text.split())
            
            overlap = len(goal_words.intersection(repo_words))
            match_score = overlap / max(len(goal_words), 1)
            match_scores.append(match_score)
        
        return sum(match_scores) / len(match_scores) if match_scores else 0.0
