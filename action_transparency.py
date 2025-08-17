"""Action Transparency System for real-time RAG communication."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from models import GitHubDiscoveryAction
from enhanced_purpose_engine import EnhancedRAGPurpose


@dataclass
class SearchAction:
    """Details of a GitHub search action"""
    search_query: str
    search_type: str  # 'repositories', 'code', 'issues'
    timestamp: datetime
    reasoning: str
    expected_results: int
    filters_applied: Dict[str, Any]
    purpose_alignment: str


@dataclass
class RepositoryAction:
    """Details of repository analysis action"""
    repository_name: str
    analysis_type: str  # 'overview', 'auth_patterns', 'implementation_details'
    timestamp: datetime
    reasoning: str
    target_files: List[str]
    analysis_depth: str  # 'shallow', 'medium', 'deep'
    success_criteria: Dict[str, Any]


@dataclass
class OutcomeAssessment:
    """Assessment of action outcome against expectations"""
    action_id: str
    expected_outcome: str
    actual_outcome: str
    success_score: float  # 0.0 - 1.0
    gap_analysis: str
    lessons_learned: List[str]
    improvement_suggestions: List[str]


class TransparentCommunicator:
    """Real-time communication system for RAG actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.active_actions = {}
        
    def announce_purpose_decision(self, purpose: EnhancedRAGPurpose) -> None:
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
        
    def announce_search_action(self, action: SearchAction) -> None:
        """Announce a search action with context"""
        
        print(f"\n🔍 EXECUTING SEARCH:")
        print(f"   🎯 Query: '{action.search_query}'")
        print(f"   📋 Type: {action.search_type}")
        print(f"   💭 Reasoning: {action.reasoning}")
        print(f"   🎪 Expected: ~{action.expected_results} results")
        
        if action.filters_applied:
            print(f"   🔧 Filters: {', '.join(f'{k}={v}' for k, v in action.filters_applied.items())}")
    
    def announce_repository_analysis(self, action: RepositoryAction) -> None:
        """Announce repository analysis action"""
        
        print(f"\n📂 ANALYZING REPOSITORY:")
        print(f"   📦 Repository: {action.repository_name}")
        print(f"   🔍 Analysis: {action.analysis_type}")
        print(f"   💭 Reasoning: {action.reasoning}")
        print(f"   📄 Target Files: {', '.join(action.target_files[:5])}")
        print(f"   🎯 Depth: {action.analysis_depth}")
    
    def announce_discovery_results(self, discovered_repos: List[GitHubDiscoveryAction], purpose: EnhancedRAGPurpose) -> None:
        """Announce discovery results with assessment"""
        
        print(f"\n✅ DISCOVERY COMPLETE:")
        print(f"   📦 Found {len(discovered_repos)} repositories")
        
        if len(discovered_repos) >= purpose.success_criteria.get('minimum_repositories', 2):
            print(f"   ✅ Met minimum repository requirement")
        else:
            print(f"   ⚠️  Below minimum repository requirement")
        
        # Show top repositories
        for i, repo in enumerate(discovered_repos[:3], 1):
            print(f"   {i}. {repo.repository_name} (⭐{repo.repository_stats.get('stars', 0)} | 📊{repo.relevance_score:.2f})")
    
    def announce_outcome_assessment(self, assessment: OutcomeAssessment) -> None:
        """Announce outcome assessment"""
        
        print(f"\n📊 ACTION ASSESSMENT:")
        print(f"   🎯 Expected: {assessment.expected_outcome}")
        print(f"   ✅ Actual: {assessment.actual_outcome}")
        print(f"   📊 Success: {assessment.success_score:.2f}")
        
        if assessment.success_score < 0.6:
            print(f"   ⚠️  Gap: {assessment.gap_analysis}")
            print(f"   💡 Suggestions: {', '.join(assessment.improvement_suggestions[:2])}")


class RAGActionTracker:
    """Comprehensive action tracking system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.communicator = TransparentCommunicator()
        self.search_actions: List[SearchAction] = []
        self.repository_actions: List[RepositoryAction] = []
        self.outcome_assessments: List[OutcomeAssessment] = []
        
    def track_search_action(self, query: str, search_type: str, purpose: EnhancedRAGPurpose, **kwargs) -> SearchAction:
        """Track a GitHub search action with context"""
        
        # Determine reasoning based on purpose
        reasoning = self.generate_search_reasoning(query, search_type, purpose)
        
        # Estimate expected results
        expected_results = self.estimate_search_results(query, search_type, purpose)
        
        # Extract filters
        filters = {k: v for k, v in kwargs.items() if k in ['language', 'sort', 'order', 'per_page']}
        
        action = SearchAction(
            search_query=query,
            search_type=search_type,
            timestamp=datetime.now(),
            reasoning=reasoning,
            expected_results=expected_results,
            filters_applied=filters,
            purpose_alignment=f"Searching for {purpose.primary_purpose}"
        )
        
        self.search_actions.append(action)
        self.communicator.announce_search_action(action)
        
        return action
    
    def track_repository_action(self, repo_name: str, analysis_type: str, purpose: EnhancedRAGPurpose, target_files: List[str] = None) -> RepositoryAction:
        """Track a repository analysis action"""
        
        # Generate reasoning
        reasoning = self.generate_repository_reasoning(repo_name, analysis_type, purpose)
        
        # Determine analysis depth
        analysis_depth = self.determine_analysis_depth(analysis_type, purpose)
        
        # Create success criteria
        success_criteria = self.create_repository_success_criteria(analysis_type, purpose)
        
        action = RepositoryAction(
            repository_name=repo_name,
            analysis_type=analysis_type,
            timestamp=datetime.now(),
            reasoning=reasoning,
            target_files=target_files or [],
            analysis_depth=analysis_depth,
            success_criteria=success_criteria
        )
        
        self.repository_actions.append(action)
        self.communicator.announce_repository_analysis(action)
        
        return action
    
    def assess_action_outcome(self, action_id: str, expected: str, actual: str, context: Dict[str, Any] = None) -> OutcomeAssessment:
        """Assess the outcome of an action against expectations"""
        
        # Calculate success score
        success_score = self.calculate_success_score(expected, actual, context or {})
        
        # Generate gap analysis
        gap_analysis = self.analyze_expectation_gap(expected, actual, success_score)
        
        # Extract lessons learned
        lessons_learned = self.extract_lessons_learned(expected, actual, context or {})
        
        # Generate improvement suggestions
        improvement_suggestions = self.generate_improvement_suggestions(expected, actual, success_score)
        
        assessment = OutcomeAssessment(
            action_id=action_id,
            expected_outcome=expected,
            actual_outcome=actual,
            success_score=success_score,
            gap_analysis=gap_analysis,
            lessons_learned=lessons_learned,
            improvement_suggestions=improvement_suggestions
        )
        
        self.outcome_assessments.append(assessment)
        self.communicator.announce_outcome_assessment(assessment)
        
        return assessment
    
    def generate_search_reasoning(self, query: str, search_type: str, purpose: EnhancedRAGPurpose) -> str:
        """Generate reasoning for search action"""
        
        if search_type == 'repositories':
            return f"Searching repositories for '{query}' to find {purpose.primary_purpose} examples in {', '.join(purpose.technologies[:2])}"
        elif search_type == 'code':
            return f"Searching code for '{query}' to find specific implementation patterns"
        else:
            return f"Searching {search_type} for '{query}' related to {purpose.primary_purpose}"
    
    def estimate_search_results(self, query: str, search_type: str, purpose: EnhancedRAGPurpose) -> int:
        """Estimate expected search results"""
        
        base_estimate = 50
        
        # Adjust based on technology popularity
        popular_techs = ['react', 'javascript', 'python', 'node']
        if any(tech in purpose.technologies for tech in popular_techs):
            base_estimate *= 2
        
        # Adjust based on query specificity
        if len(query.split()) > 3:
            base_estimate = int(base_estimate * 0.7)  # More specific = fewer results
        
        # Adjust based on search type
        if search_type == 'code':
            base_estimate *= 3  # Code searches return more results
        
        return min(base_estimate, 200)  # Cap at reasonable number
    
    def generate_repository_reasoning(self, repo_name: str, analysis_type: str, purpose: EnhancedRAGPurpose) -> str:
        """Generate reasoning for repository analysis"""
        
        if analysis_type == 'overview':
            return f"Analyzing {repo_name} overview to assess relevance for {purpose.primary_purpose}"
        elif analysis_type == 'auth_patterns':
            return f"Deep-diving into {repo_name} authentication patterns for implementation guidance"
        elif analysis_type == 'implementation_details':
            return f"Extracting specific implementation details from {repo_name} for solution development"
        else:
            return f"Analyzing {repo_name} for {analysis_type} related to {purpose.primary_purpose}"
    
    def determine_analysis_depth(self, analysis_type: str, purpose: EnhancedRAGPurpose) -> str:
        """Determine appropriate analysis depth"""
        
        if purpose.reasoning and purpose.reasoning.technical_complexity > 7:
            return 'deep'
        elif analysis_type in ['implementation_details', 'auth_patterns']:
            return 'medium'
        else:
            return 'shallow'
    
    def create_repository_success_criteria(self, analysis_type: str, purpose: EnhancedRAGPurpose) -> Dict[str, Any]:
        """Create success criteria for repository analysis"""
        
        return {
            'find_relevant_files': len(purpose.expected_file_patterns),
            'extract_code_examples': analysis_type in ['auth_patterns', 'implementation_details'],
            'identify_best_practices': purpose.reasoning.technical_complexity > 5 if purpose.reasoning else False,
            'assess_production_readiness': purpose.urgency_context and 'production' in purpose.urgency_context.lower()
        }
    
    def calculate_success_score(self, expected: str, actual: str, context: Dict[str, Any]) -> float:
        """Calculate success score for an outcome"""
        
        score = 0.5  # Base score
        
        # Simple keyword matching for now - could be enhanced with semantic similarity
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        overlap = len(expected_words.intersection(actual_words))
        total_expected = len(expected_words)
        
        if total_expected > 0:
            keyword_score = overlap / total_expected
            score = (score + keyword_score) / 2
        
        # Adjust based on context
        if context.get('found_repositories', 0) > 0:
            score += 0.2
        if context.get('found_code_examples', 0) > 0:
            score += 0.2
        if context.get('relevance_score', 0) > 0.7:
            score += 0.1
        
        return min(score, 1.0)
    
    def analyze_expectation_gap(self, expected: str, actual: str, success_score: float) -> str:
        """Analyze the gap between expectations and reality"""
        
        if success_score > 0.8:
            return "Expectations well met"
        elif success_score > 0.6:
            return "Expectations partially met - some gaps in coverage"
        elif success_score > 0.4:
            return "Significant gaps between expectations and results"
        else:
            return "Major disconnect between expectations and reality"
    
    def extract_lessons_learned(self, expected: str, actual: str, context: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from the outcome"""
        
        lessons = []
        
        if context.get('found_repositories', 0) == 0:
            lessons.append("Search terms may be too specific - consider broader queries")
        
        if context.get('relevance_score', 0) < 0.5:
            lessons.append("Repository relevance is low - refine search criteria")
        
        if 'no code examples' in actual.lower():
            lessons.append("Repositories lack practical implementation examples")
        
        if not lessons:
            lessons.append("Search strategy was effective")
        
        return lessons
    
    def generate_improvement_suggestions(self, expected: str, actual: str, success_score: float) -> List[str]:
        """Generate improvement suggestions"""
        
        suggestions = []
        
        if success_score < 0.6:
            suggestions.append("Refine search queries with more specific technical terms")
            suggestions.append("Try alternative technology combinations")
        
        if 'no recent updates' in actual.lower():
            suggestions.append("Focus on more recently updated repositories")
        
        if success_score < 0.4:
            suggestions.append("Consider expanding search to include related technologies")
            suggestions.append("Look for community tutorials and documentation")
        
        return suggestions[:3]  # Top 3 suggestions
    
    def get_action_summary(self) -> Dict[str, Any]:
        """Get comprehensive action summary"""
        
        return {
            'total_search_actions': len(self.search_actions),
            'total_repository_actions': len(self.repository_actions),
            'total_assessments': len(self.outcome_assessments),
            'average_success_score': sum(a.success_score for a in self.outcome_assessments) / len(self.outcome_assessments) if self.outcome_assessments else 0,
            'recent_actions': {
                'searches': [asdict(action) for action in self.search_actions[-3:]],
                'repositories': [asdict(action) for action in self.repository_actions[-3:]],
                'assessments': [asdict(assessment) for assessment in self.outcome_assessments[-3:]]
            }
        }
