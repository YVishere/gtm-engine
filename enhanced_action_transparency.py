"""Enhanced action transparency with LLM-driven decision tracking."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from models import GitHubDiscoveryAction
from enhanced_purpose_engine import EnhancedRAGPurpose
from llm_search_strategist import LLMSearchStrategy, RepositoryAnalysisStrategy, APIRateLimit


@dataclass
class LLMDecisionRecord:
    """Record of an LLM-driven decision"""
    decision_type: str  # 'search_strategy', 'repository_analysis', 'outcome_assessment'
    timestamp: datetime
    context_provided: Dict[str, Any]
    llm_response: Dict[str, Any]
    confidence_score: float
    reasoning: str
    api_usage_estimate: int
    actual_api_usage: int = 0
    success_score: float = 0.0  # Set after execution


@dataclass
class SearchExecutionResult:
    """Result of executing a search strategy"""
    strategy: LLMSearchStrategy
    queries_executed: List[Dict[str, Any]]
    repositories_found: List[Dict[str, Any]]
    api_requests_used: int
    execution_time: float
    success_metrics: Dict[str, Any]
    gaps_identified: List[str]


@dataclass
class RepositoryAnalysisResult:
    """Result of executing repository analysis"""
    strategy: RepositoryAnalysisStrategy
    repositories_analyzed: List[str]
    files_examined: List[str]
    code_patterns_found: List[str]
    api_requests_used: int
    analysis_quality: float
    insights_extracted: List[str]


class EnhancedTransparentCommunicator:
    """Enhanced real-time communication system for LLM-driven actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def announce_new_opportunity(self, opportunity_number: int) -> None:
        """Announce the start of a new opportunity with fresh API budget"""
        print(f"\n🎯 STARTING NEW OPPORTUNITY #{opportunity_number}")
        print(f"   🔄 Fresh API budget: 7 requests available")
        print(f"   🎪 Ready to explore and discover!")
    
    def announce_purpose_decision(self, purpose) -> None:
        """Announce the purpose detection decision with reasoning"""
        
        print(f"\n🎯 RAG PURPOSE DETECTED:")
        print(f"   📝 Goal: {purpose.primary_purpose}")
        print(f"   🔧 Technologies: {', '.join(purpose.technologies)}")
        print(f"   📋 Strategy: {purpose.search_strategy}")
        print(f"   ⚡ Context: {purpose.urgency_context}")
        print(f"   🎪 Confidence: {purpose.confidence_score:.2f}")
        
        if purpose.reasoning:
            print(f"   🧠 Problem Type: {purpose.reasoning.problem_type}")
            print(f"   🎯 Complexity: {purpose.reasoning.technical_complexity}/10")
            print(f"   📊 Requirements: {', '.join(purpose.reasoning.solution_requirements[:3])}")
        
        print(f"   🔮 Expected Results:")
        for i, repo_type in enumerate(purpose.expected_repositories[:3], 1):
            print(f"      {i}. {repo_type}")
        
    def announce_llm_decision(self, decision: LLMDecisionRecord) -> None:
        """Announce LLM decision with full transparency"""
        
        print(f"\n🧠 LLM DECISION: {decision.decision_type.upper()}")
        print(f"   🎯 Reasoning: {decision.reasoning}")
        print(f"   📊 Confidence: {decision.confidence_score:.2f}")
        print(f"   🔧 API Budget: {decision.api_usage_estimate} requests")
        print(f"   ⏰ Timestamp: {decision.timestamp.strftime('%H:%M:%S')}")
        
        if decision.decision_type == 'search_strategy':
            self._announce_search_strategy_decision(decision)
        elif decision.decision_type == 'repository_analysis':
            self._announce_repository_analysis_decision(decision)
    
    def announce_rate_limit_status(self, rate_limit: APIRateLimit) -> None:
        """Announce current API rate limit status"""
        
        remaining_opportunity = rate_limit.get_remaining_for_opportunity()
        remaining_session = rate_limit.get_remaining_for_session()
        opportunity_percentage = (rate_limit.requests_used_current_opportunity / rate_limit.max_requests_per_opportunity) * 100
        session_percentage = (rate_limit.requests_used_session / rate_limit.max_requests_per_session) * 100
        
        print(f"\n📊 API RATE LIMIT STATUS:")
        print(f"   🎯 OPPORTUNITY {rate_limit.current_opportunity_index + 1}:")
        print(f"      🔢 Used: {rate_limit.requests_used_current_opportunity}/{rate_limit.max_requests_per_opportunity}")
        print(f"      ⚡ Remaining: {remaining_opportunity} requests")
        print(f"      📈 Usage: {opportunity_percentage:.1f}%")
        
        print(f"   🔄 SESSION TOTAL:")
        print(f"      🔢 Used: {rate_limit.requests_used_session}/{rate_limit.max_requests_per_session}")
        print(f"      ⚡ Remaining: {remaining_session} requests")
        print(f"      📈 Usage: {session_percentage:.1f}%")
        
        if remaining_opportunity <= 1:
            print(f"   🚨 CRITICAL: Opportunity budget nearly exhausted!")
        elif remaining_opportunity <= 3:
            print(f"   ⚠️  WARNING: Low opportunity requests remaining!")
        elif remaining_session <= 10:
            print(f"   🟡 CAUTION: Session requests running low")
        else:
            print(f"   ✅ GOOD: Sufficient API requests available")
    
    def announce_search_execution(self, result: SearchExecutionResult) -> None:
        """Announce search execution results"""
        
        print(f"\n🔍 SEARCH EXECUTION COMPLETE:")
        print(f"   📋 Queries Run: {len(result.queries_executed)}")
        print(f"   📦 Repositories Found: {len(result.repositories_found)}")
        print(f"   🔧 API Requests Used: {result.api_requests_used}")
        print(f"   ⏱️  Execution Time: {result.execution_time:.2f}s")
        
        # Show success metrics
        metrics = result.success_metrics
        if metrics.get('target_met', False):
            print(f"   ✅ Target Met: Found {metrics.get('repositories_found', 0)} repositories")
        else:
            print(f"   ⚠️  Target Missed: Expected {metrics.get('expected', 0)}, found {metrics.get('actual', 0)}")
        
        # Show top repositories
        top_repos = sorted(result.repositories_found, 
                          key=lambda r: r.get('stargazers_count', 0), 
                          reverse=True)[:3]
        
        print(f"   🌟 Top Repositories:")
        for i, repo in enumerate(top_repos, 1):
            print(f"      {i}. {repo.get('full_name', 'unknown')} (⭐{repo.get('stargazers_count', 0)})")
        
        # Show any gaps identified
        if result.gaps_identified:
            print(f"   🔍 Gaps Identified:")
            for gap in result.gaps_identified[:2]:
                print(f"      • {gap}")
    
    def announce_repository_analysis(self, result: RepositoryAnalysisResult) -> None:
        """Announce repository analysis results"""
        
        print(f"\n📂 REPOSITORY ANALYSIS COMPLETE:")
        print(f"   📦 Repositories Analyzed: {len(result.repositories_analyzed)}")
        print(f"   📄 Files Examined: {len(result.files_examined)}")
        print(f"   🔧 API Requests Used: {result.api_requests_used}")
        print(f"   📊 Analysis Quality: {result.analysis_quality:.2f}")
        
        # Show patterns found
        if result.code_patterns_found:
            print(f"   🎯 Key Patterns Found:")
            for pattern in result.code_patterns_found[:3]:
                print(f"      • {pattern}")
        
        # Show insights
        if result.insights_extracted:
            print(f"   💡 Insights Extracted:")
            for insight in result.insights_extracted[:2]:
                print(f"      • {insight}")
    
    def announce_llm_outcome_assessment(self, assessment: Dict[str, Any]) -> None:
        """Announce LLM-generated outcome assessment"""
        
        print(f"\n🎯 LLM OUTCOME ASSESSMENT:")
        print(f"   📊 Overall Success: {assessment.get('overall_success_score', 0):.2f}")
        print(f"   ✅ Strengths: {assessment.get('strengths', 'Not specified')}")
        print(f"   ⚠️  Weaknesses: {assessment.get('weaknesses', 'Not specified')}")
        print(f"   💡 Recommendations: {assessment.get('recommendations', 'Not specified')}")
        
        if assessment.get('lessons_learned'):
            print(f"   🎓 Lessons Learned:")
            for lesson in assessment.get('lessons_learned', [])[:2]:
                print(f"      • {lesson}")
    
    def _announce_search_strategy_decision(self, decision: LLMDecisionRecord) -> None:
        """Announce search strategy specific details"""
        
        strategy_data = decision.llm_response
        
        if 'search_queries' in strategy_data:
            print(f"   🔍 Search Queries Planned:")
            for i, query in enumerate(strategy_data['search_queries'][:3], 1):
                print(f"      {i}. '{query.get('query', 'unknown')}' ({query.get('priority', 'medium')} priority)")
        
        if 'expected_outcomes' in strategy_data:
            outcomes = strategy_data['expected_outcomes']
            print(f"   🎯 Expected: {outcomes.get('total_repositories', 'unknown')} repositories")
    
    def _announce_repository_analysis_decision(self, decision: LLMDecisionRecord) -> None:
        """Announce repository analysis specific details"""
        
        strategy_data = decision.llm_response
        
        if 'target_repositories' in strategy_data:
            print(f"   📦 Target Repositories:")
            for repo in strategy_data['target_repositories'][:3]:
                print(f"      • {repo}")
        
        if 'analysis_depth' in strategy_data:
            print(f"   🔍 Analysis Depth: {strategy_data['analysis_depth']}")


class LLMDrivenActionTracker:
    """Enhanced action tracking with LLM decision recording"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.communicator = EnhancedTransparentCommunicator()
        self.llm_decisions: List[LLMDecisionRecord] = []
        self.search_executions: List[SearchExecutionResult] = []
        self.repository_analyses: List[RepositoryAnalysisResult] = []
        self.rate_limit_tracker = APIRateLimit()
        
        # Backward compatibility attributes
        self.search_actions = []  # For compatibility with existing code
        self.repository_actions = []
        self.outcome_assessments = []
    
    def start_new_opportunity(self) -> None:
        """Start tracking a new opportunity with fresh API budget"""
        self.rate_limit_tracker.start_new_opportunity()
        self.communicator.announce_new_opportunity(self.rate_limit_tracker.current_opportunity_index + 1)
        
    def record_llm_decision(self, 
                           decision_type: str,
                           context: Dict[str, Any],
                           llm_response: Dict[str, Any],
                           reasoning: str,
                           confidence: float,
                           api_estimate: int) -> LLMDecisionRecord:
        """Record an LLM decision with full context"""
        
        decision = LLMDecisionRecord(
            decision_type=decision_type,
            timestamp=datetime.now(),
            context_provided=context,
            llm_response=llm_response,
            confidence_score=confidence,
            reasoning=reasoning,
            api_usage_estimate=api_estimate
        )
        
        self.llm_decisions.append(decision)
        self.communicator.announce_llm_decision(decision)
        
        return decision
    
    def record_search_execution(self, result: SearchExecutionResult) -> None:
        """Record search execution results"""
        
        self.search_executions.append(result)
        self.rate_limit_tracker.requests_used_current_opportunity += result.api_requests_used
        self.rate_limit_tracker.requests_used_session += result.api_requests_used
        
        # Update the corresponding LLM decision with actual results
        self._update_decision_with_results('search_strategy', result.api_requests_used, 
                                         result.success_metrics.get('success_score', 0.5))
        
        self.communicator.announce_search_execution(result)
        self.communicator.announce_rate_limit_status(self.rate_limit_tracker)
    
    def record_repository_analysis(self, result: RepositoryAnalysisResult) -> None:
        """Record repository analysis results"""
        
        self.repository_analyses.append(result)
        self.rate_limit_tracker.requests_used_current_opportunity += result.api_requests_used
        self.rate_limit_tracker.requests_used_session += result.api_requests_used
        
        # Update the corresponding LLM decision with actual results
        self._update_decision_with_results('repository_analysis', result.api_requests_used, 
                                         result.analysis_quality)
        
        self.communicator.announce_repository_analysis(result)
        self.communicator.announce_rate_limit_status(self.rate_limit_tracker)
    
    def can_proceed_with_api_usage(self, estimated_requests: int) -> bool:
        """Check if we can proceed with estimated API usage"""
        
        remaining_opportunity = self.rate_limit_tracker.get_remaining_for_opportunity()
        remaining_session = self.rate_limit_tracker.get_remaining_for_session()
        
        if remaining_opportunity < estimated_requests:
            self.logger.warning(f"Insufficient opportunity API requests: need {estimated_requests}, have {remaining_opportunity}")
            return False
        elif remaining_session < estimated_requests:
            self.logger.warning(f"Insufficient session API requests: need {estimated_requests}, have {remaining_session}")
            return False
        
        return True
    
    def track_api_usage(self, requests_used: int) -> bool:
        """Track API usage for the current opportunity and check if we can continue"""
        
        self.rate_limit_tracker.requests_used_current_opportunity += requests_used
        self.rate_limit_tracker.requests_used_session += requests_used
        self.rate_limit_tracker.last_request_time = datetime.now()
        
        remaining_opportunity = self.rate_limit_tracker.get_remaining_for_opportunity()
        remaining_session = self.rate_limit_tracker.get_remaining_for_session()
        
        print(f"\n📈 TRACKING API USAGE:")
        print(f"   🎯 Opportunity {self.rate_limit_tracker.current_opportunity_index + 1}: +{requests_used} requests")
        print(f"   📊 Opportunity Total: {self.rate_limit_tracker.requests_used_current_opportunity}/{self.rate_limit_tracker.max_requests_per_opportunity}")
        print(f"   🔄 Session Total: {self.rate_limit_tracker.requests_used_session}/{self.rate_limit_tracker.max_requests_per_session}")
        
        if remaining_opportunity <= 0:
            self.logger.warning("API rate limit reached for this opportunity")
            return False
        elif remaining_session <= 0:
            self.logger.warning("API rate limit reached for this session")
            return False
        elif remaining_opportunity <= 1:
            self.logger.warning(f"Opportunity API rate limit nearly reached: {remaining_opportunity} requests remaining")
        
        return True
    
    def get_api_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive API usage summary"""
        
        total_decisions = len(self.llm_decisions)
        total_searches = len(self.search_executions)
        total_analyses = len(self.repository_analyses)
        
        return {
            "total_api_requests_used": self.rate_limit_tracker.requests_used_session,
            "max_requests_allowed": self.rate_limit_tracker.max_requests_per_session,
            "remaining_requests": self.rate_limit_tracker.get_remaining_for_session(),
            "usage_percentage": (self.rate_limit_tracker.requests_used_session / self.rate_limit_tracker.max_requests_per_session) * 100,
            "llm_decisions_made": total_decisions,
            "search_executions": total_searches,
            "repository_analyses": total_analyses,
            "average_confidence": sum(d.confidence_score for d in self.llm_decisions) / max(total_decisions, 1),
            "session_duration": (datetime.now() - self.rate_limit_tracker.session_start).total_seconds() / 60,
            "requests_per_minute": self.rate_limit_tracker.requests_used_session / max((datetime.now() - self.rate_limit_tracker.session_start).total_seconds() / 60, 1)
        }
    
    def generate_session_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive session analytics"""
        
        analytics = {
            "session_summary": self.get_api_usage_summary(),
            "llm_decision_analysis": self._analyze_llm_decisions(),
            "search_effectiveness": self._analyze_search_effectiveness(),
            "repository_analysis_quality": self._analyze_repository_quality(),
            "recommendations": self._generate_recommendations()
        }
        
        return analytics
    
    def _update_decision_with_results(self, decision_type: str, actual_api: int, success_score: float) -> None:
        """Update LLM decision with actual execution results"""
        
        # Find the most recent decision of this type
        for decision in reversed(self.llm_decisions):
            if decision.decision_type == decision_type and decision.actual_api_usage == 0:
                decision.actual_api_usage = actual_api
                decision.success_score = success_score
                break
    
    def _analyze_llm_decisions(self) -> Dict[str, Any]:
        """Analyze LLM decision quality"""
        
        if not self.llm_decisions:
            return {"analysis": "No LLM decisions recorded"}
        
        total_decisions = len(self.llm_decisions)
        avg_confidence = sum(d.confidence_score for d in self.llm_decisions) / total_decisions
        avg_success = sum(d.success_score for d in self.llm_decisions if d.success_score > 0) / max(sum(1 for d in self.llm_decisions if d.success_score > 0), 1)
        
        api_accuracy = []
        for decision in self.llm_decisions:
            if decision.actual_api_usage > 0:
                accuracy = 1 - abs(decision.api_usage_estimate - decision.actual_api_usage) / max(decision.api_usage_estimate, 1)
                api_accuracy.append(max(accuracy, 0))
        
        avg_api_accuracy = sum(api_accuracy) / max(len(api_accuracy), 1)
        
        return {
            "total_decisions": total_decisions,
            "average_confidence": avg_confidence,
            "average_success_score": avg_success,
            "api_estimation_accuracy": avg_api_accuracy,
            "decision_types": {dt: sum(1 for d in self.llm_decisions if d.decision_type == dt) 
                             for dt in set(d.decision_type for d in self.llm_decisions)}
        }
    
    def _analyze_search_effectiveness(self) -> Dict[str, Any]:
        """Analyze search execution effectiveness"""
        
        if not self.search_executions:
            return {"analysis": "No search executions recorded"}
        
        total_repos = sum(len(s.repositories_found) for s in self.search_executions)
        total_api_used = sum(s.api_requests_used for s in self.search_executions)
        avg_repos_per_api = total_repos / max(total_api_used, 1)
        
        return {
            "total_searches": len(self.search_executions),
            "total_repositories_found": total_repos,
            "total_api_requests": total_api_used,
            "efficiency": avg_repos_per_api,
            "average_execution_time": sum(s.execution_time for s in self.search_executions) / len(self.search_executions)
        }
    
    def _analyze_repository_quality(self) -> Dict[str, Any]:
        """Analyze repository analysis quality"""
        
        if not self.repository_analyses:
            return {"analysis": "No repository analyses recorded"}
        
        total_repos = sum(len(r.repositories_analyzed) for r in self.repository_analyses)
        avg_quality = sum(r.analysis_quality for r in self.repository_analyses) / len(self.repository_analyses)
        total_insights = sum(len(r.insights_extracted) for r in self.repository_analyses)
        
        return {
            "total_analyses": len(self.repository_analyses),
            "repositories_analyzed": total_repos,
            "average_quality_score": avg_quality,
            "total_insights_extracted": total_insights,
            "insights_per_repository": total_insights / max(total_repos, 1)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on session analytics"""
        
        recommendations = []
        
        # API usage recommendations
        usage_pct = (self.rate_limit_tracker.requests_used_session / self.rate_limit_tracker.max_requests_per_session) * 100
        if usage_pct > 80:
            recommendations.append("Consider increasing API request limits for future sessions")
        elif usage_pct < 30:
            recommendations.append("API requests were underutilized - could explore more repositories")
        
        # LLM decision recommendations
        if self.llm_decisions:
            avg_confidence = sum(d.confidence_score for d in self.llm_decisions) / len(self.llm_decisions)
            if avg_confidence < 0.7:
                recommendations.append("LLM confidence was low - consider providing more context")
        
        # Search effectiveness recommendations
        if self.search_executions:
            total_repos = sum(len(s.repositories_found) for s in self.search_executions)
            if total_repos < 10:
                recommendations.append("Consider broader search queries to find more repositories")
            elif total_repos > 50:
                recommendations.append("Consider more specific search queries to reduce noise")
        
        return recommendations
