"""Enhanced Reporting System for comprehensive RAG analytics."""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from models import EmailSolution, GitHubDiscoveryAction
from enhanced_purpose_engine import EnhancedRAGPurpose, OpportunityAnalysis
from action_transparency import RAGActionTracker, OutcomeAssessment


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class SessionAnalytics:
    """Analytics for a complete RAG session"""
    session_id: str
    start_time: datetime
    end_time: datetime
    total_opportunities: int
    total_searches: int
    total_repositories_analyzed: int
    total_emails_generated: int
    average_confidence: float
    average_success_score: float
    technology_coverage: Dict[str, int]
    problem_type_distribution: Dict[str, int]
    quality_metrics: Dict[str, float]


@dataclass
class PerformanceMetrics:
    """Performance metrics for RAG operations"""
    search_efficiency: float  # repos found / searches made
    analysis_depth_score: float  # average analysis quality
    solution_relevance: float  # average email relevance
    technology_match_rate: float  # how well tech detection worked
    purpose_accuracy: float  # purpose detection accuracy
    time_per_opportunity: float  # average processing time


@dataclass
class QualityInsight:
    """Quality insight from pattern analysis"""
    insight_type: str  # 'improvement', 'pattern', 'anomaly'
    category: str  # 'search', 'analysis', 'generation'
    description: str
    impact_level: str  # 'high', 'medium', 'low'
    suggested_action: str
    evidence: List[str]


@dataclass
class LearningPattern:
    """Detected learning pattern for improvement"""
    pattern_id: str
    pattern_type: str  # 'technology_preference', 'search_strategy', 'quality_correlation'
    description: str
    frequency: int
    confidence: float
    recommendation: str


class RAGSessionLogger:
    """Comprehensive session logging system"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"rag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_time = datetime.now()
        self.events: List[Dict[str, Any]] = []
        self.session_data: Dict[str, Any] = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'opportunities': [],
            'actions': [],
            'outcomes': [],
            'emails': []
        }
        
    def log_opportunity_processing(self, opportunity_index: int, purpose: EnhancedRAGPurpose) -> None:
        """Log opportunity processing start"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'opportunity_start',
            'opportunity_index': opportunity_index,
            'purpose': {
                'primary_purpose': purpose.primary_purpose,
                'technologies': purpose.technologies,
                'confidence_score': purpose.confidence_score,
                'complexity': purpose.reasoning.technical_complexity if purpose.reasoning else 0
            }
        }
        
        self.events.append(event)
        self.session_data['opportunities'].append(event)
        
    def log_action_execution(self, action_type: str, details: Dict[str, Any]) -> None:
        """Log action execution"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'action_execution',
            'action_type': action_type,
            'details': details
        }
        
        self.events.append(event)
        self.session_data['actions'].append(event)
        
    def log_outcome_assessment(self, assessment: OutcomeAssessment) -> None:
        """Log outcome assessment"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'outcome_assessment',
            'assessment': asdict(assessment)
        }
        
        self.events.append(event)
        self.session_data['outcomes'].append(event)
        
    def log_email_generation(self, email_index: int, email_solution: EmailSolution) -> None:
        """Log email generation completion"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'email_generated',
            'email_index': email_index,
            'email_data': {
                'original_query': email_solution.original_query,
                'solution_quality': email_solution.solution_quality,
                'github_actions_count': len(email_solution.github_actions),
                'confidence_score': email_solution.confidence_score
            }
        }
        
        self.events.append(event)
        self.session_data['emails'].append(event)
        
    def finalize_session(self) -> str:
        """Finalize session and save log file"""
        
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['total_events'] = len(self.events)
        self.session_data['events'] = self.events
        
        # Create logs directory if it doesn't exist
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Save session log
        log_file = logs_dir / f"{self.session_id}.json"
        with open(log_file, 'w') as f:
            json.dump(self.session_data, f, indent=2, cls=DateTimeEncoder)
            
        return str(log_file)


class RAGPerformanceAnalyzer:
    """Performance analysis for RAG operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_session_performance(self, session_data: Dict[str, Any], action_tracker: RAGActionTracker) -> PerformanceMetrics:
        """Analyze performance metrics for a session"""
        
        # Calculate search efficiency
        search_efficiency = self.calculate_search_efficiency(action_tracker)
        
        # Calculate analysis depth score
        analysis_depth_score = self.calculate_analysis_depth_score(action_tracker)
        
        # Calculate solution relevance
        solution_relevance = self.calculate_solution_relevance(session_data)
        
        # Calculate technology match rate
        technology_match_rate = self.calculate_technology_match_rate(session_data)
        
        # Calculate purpose accuracy
        purpose_accuracy = self.calculate_purpose_accuracy(action_tracker)
        
        # Calculate time per opportunity
        time_per_opportunity = self.calculate_time_per_opportunity(session_data)
        
        return PerformanceMetrics(
            search_efficiency=search_efficiency,
            analysis_depth_score=analysis_depth_score,
            solution_relevance=solution_relevance,
            technology_match_rate=technology_match_rate,
            purpose_accuracy=purpose_accuracy,
            time_per_opportunity=time_per_opportunity
        )
    
    def calculate_search_efficiency(self, action_tracker: RAGActionTracker) -> float:
        """Calculate how efficiently searches find relevant repositories"""
        
        if not action_tracker.search_actions:
            return 0.0
            
        total_searches = len(action_tracker.search_actions)
        successful_assessments = [a for a in action_tracker.outcome_assessments if a.success_score > 0.6]
        
        return len(successful_assessments) / total_searches if total_searches > 0 else 0.0
    
    def calculate_analysis_depth_score(self, action_tracker: RAGActionTracker) -> float:
        """Calculate the quality of repository analysis"""
        
        if not action_tracker.repository_actions:
            return 0.0
            
        depth_scores = {
            'shallow': 0.3,
            'medium': 0.6,
            'deep': 1.0
        }
        
        total_score = sum(depth_scores.get(action.analysis_depth, 0.5) for action in action_tracker.repository_actions)
        return total_score / len(action_tracker.repository_actions)
    
    def calculate_solution_relevance(self, session_data: Dict[str, Any]) -> float:
        """Calculate average solution relevance from emails"""
        
        emails = session_data.get('emails', [])
        if not emails:
            return 0.0
            
        relevance_scores = [email['email_data'].get('confidence_score', 0.5) for email in emails]
        return sum(relevance_scores) / len(relevance_scores)
    
    def calculate_technology_match_rate(self, session_data: Dict[str, Any]) -> float:
        """Calculate how well technology detection worked"""
        
        opportunities = session_data.get('opportunities', [])
        if not opportunities:
            return 0.0
            
        # Simple heuristic: opportunities with more technologies detected score higher
        tech_counts = [len(opp['purpose'].get('technologies', [])) for opp in opportunities]
        avg_tech_count = sum(tech_counts) / len(tech_counts)
        
        # Normalize to 0-1 scale (assuming 3+ technologies is good)
        return min(avg_tech_count / 3.0, 1.0)
    
    def calculate_purpose_accuracy(self, action_tracker: RAGActionTracker) -> float:
        """Calculate purpose detection accuracy"""
        
        if not action_tracker.outcome_assessments:
            return 0.0
            
        purpose_related_assessments = [a for a in action_tracker.outcome_assessments if 'purpose' in a.expected_outcome.lower()]
        if not purpose_related_assessments:
            return 0.7  # Default if no purpose-specific assessments
            
        return sum(a.success_score for a in purpose_related_assessments) / len(purpose_related_assessments)
    
    def calculate_time_per_opportunity(self, session_data: Dict[str, Any]) -> float:
        """Calculate average processing time per opportunity"""
        
        start_time = datetime.fromisoformat(session_data['start_time'])
        end_time = datetime.fromisoformat(session_data.get('end_time', datetime.now().isoformat()))
        
        total_time = (end_time - start_time).total_seconds()
        opportunities_count = len(session_data.get('opportunities', []))
        
        return total_time / opportunities_count if opportunities_count > 0 else 0.0


class RAGPatternDetector:
    """Pattern detection for learning and improvement"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def detect_learning_patterns(self, session_data: Dict[str, Any], action_tracker: RAGActionTracker) -> List[LearningPattern]:
        """Detect learning patterns from session data"""
        
        patterns = []
        
        # Technology preference patterns
        tech_patterns = self.detect_technology_patterns(session_data)
        patterns.extend(tech_patterns)
        
        # Search strategy patterns
        search_patterns = self.detect_search_strategy_patterns(action_tracker)
        patterns.extend(search_patterns)
        
        # Quality correlation patterns
        quality_patterns = self.detect_quality_patterns(action_tracker)
        patterns.extend(quality_patterns)
        
        return patterns
    
    def detect_technology_patterns(self, session_data: Dict[str, Any]) -> List[LearningPattern]:
        """Detect technology preference patterns"""
        
        patterns = []
        
        # Count technology frequencies
        tech_counts = {}
        for opp in session_data.get('opportunities', []):
            for tech in opp['purpose'].get('technologies', []):
                tech_counts[tech] = tech_counts.get(tech, 0) + 1
        
        # Find dominant technologies
        if tech_counts:
            max_count = max(tech_counts.values())
            dominant_techs = [tech for tech, count in tech_counts.items() if count == max_count]
            
            if len(dominant_techs) == 1 and max_count > 1:
                pattern = LearningPattern(
                    pattern_id=f"tech_preference_{dominant_techs[0]}",
                    pattern_type="technology_preference",
                    description=f"Strong preference detected for {dominant_techs[0]} technology",
                    frequency=max_count,
                    confidence=min(max_count / len(session_data.get('opportunities', [])), 1.0),
                    recommendation=f"Prioritize {dominant_techs[0]}-specific repositories and examples"
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_search_strategy_patterns(self, action_tracker: RAGActionTracker) -> List[LearningPattern]:
        """Detect search strategy patterns"""
        
        patterns = []
        
        if action_tracker.search_actions:
            # Analyze query lengths
            query_lengths = [len(action.search_query.split()) for action in action_tracker.search_actions]
            avg_length = sum(query_lengths) / len(query_lengths)
            
            if avg_length > 4:
                pattern = LearningPattern(
                    pattern_id="long_query_preference",
                    pattern_type="search_strategy",
                    description="Preference for detailed, specific search queries",
                    frequency=len([l for l in query_lengths if l > 4]),
                    confidence=0.8,
                    recommendation="Continue using detailed queries but also test broader terms"
                )
                patterns.append(pattern)
            elif avg_length < 2:
                pattern = LearningPattern(
                    pattern_id="short_query_preference",
                    pattern_type="search_strategy",
                    description="Preference for short, general search queries",
                    frequency=len([l for l in query_lengths if l < 2]),
                    confidence=0.8,
                    recommendation="Consider adding more specific technical terms to queries"
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_quality_patterns(self, action_tracker: RAGActionTracker) -> List[LearningPattern]:
        """Detect quality correlation patterns"""
        
        patterns = []
        
        if action_tracker.outcome_assessments:
            high_quality_actions = [a for a in action_tracker.outcome_assessments if a.success_score > 0.8]
            
            if len(high_quality_actions) > len(action_tracker.outcome_assessments) * 0.7:
                pattern = LearningPattern(
                    pattern_id="high_quality_consistency",
                    pattern_type="quality_correlation",
                    description="Consistently high-quality outcomes across actions",
                    frequency=len(high_quality_actions),
                    confidence=0.9,
                    recommendation="Current strategy is effective - maintain approach"
                )
                patterns.append(pattern)
        
        return patterns


class RAGAnalyticsEngine:
    """Comprehensive analytics engine for RAG operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session_logger = RAGSessionLogger()
        self.performance_analyzer = RAGPerformanceAnalyzer()
        self.pattern_detector = RAGPatternDetector()
        
    def generate_comprehensive_report(self, session_data: Dict[str, Any], action_tracker: RAGActionTracker) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        
        print("\n📊 GENERATING COMPREHENSIVE RAG ANALYTICS...")
        
        # Performance analysis
        performance_metrics = self.performance_analyzer.analyze_session_performance(session_data, action_tracker)
        
        # Pattern detection
        learning_patterns = self.pattern_detector.detect_learning_patterns(session_data, action_tracker)
        
        # Quality insights
        quality_insights = self.generate_quality_insights(performance_metrics, action_tracker)
        
        # Session analytics
        session_analytics = self.compile_session_analytics(session_data, performance_metrics)
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'session_analytics': asdict(session_analytics),
            'performance_metrics': asdict(performance_metrics),
            'learning_patterns': [asdict(pattern) for pattern in learning_patterns],
            'quality_insights': [asdict(insight) for insight in quality_insights],
            'recommendations': self.generate_recommendations(performance_metrics, learning_patterns, quality_insights)
        }
        
        # Save report
        report_file = self.save_analytics_report(report)
        
        # Display summary
        self.display_analytics_summary(report)
        
        return report
    
    def generate_quality_insights(self, performance_metrics: PerformanceMetrics, action_tracker: RAGActionTracker) -> List[QualityInsight]:
        """Generate quality insights from performance data"""
        
        insights = []
        
        # Search efficiency insight
        if performance_metrics.search_efficiency < 0.5:
            insights.append(QualityInsight(
                insight_type="improvement",
                category="search",
                description="Search efficiency is below optimal - searches are not finding relevant repositories consistently",
                impact_level="high",
                suggested_action="Refine search query construction and add more specific technical terms",
                evidence=[f"Search efficiency: {performance_metrics.search_efficiency:.2f}"]
            ))
        
        # Solution relevance insight
        if performance_metrics.solution_relevance > 0.8:
            insights.append(QualityInsight(
                insight_type="pattern",
                category="generation",
                description="Email solutions consistently show high relevance scores",
                impact_level="high",
                suggested_action="Current solution generation approach is effective - maintain strategy",
                evidence=[f"Solution relevance: {performance_metrics.solution_relevance:.2f}"]
            ))
        
        # Analysis depth insight
        if performance_metrics.analysis_depth_score < 0.6:
            insights.append(QualityInsight(
                insight_type="improvement",
                category="analysis",
                description="Repository analysis could be more thorough for better solution quality",
                impact_level="medium",
                suggested_action="Increase analysis depth for high-complexity opportunities",
                evidence=[f"Analysis depth score: {performance_metrics.analysis_depth_score:.2f}"]
            ))
        
        return insights
    
    def compile_session_analytics(self, session_data: Dict[str, Any], performance_metrics: PerformanceMetrics) -> SessionAnalytics:
        """Compile comprehensive session analytics"""
        
        start_time = datetime.fromisoformat(session_data['start_time'])
        end_time = datetime.fromisoformat(session_data.get('end_time', datetime.now().isoformat()))
        
        # Technology coverage
        tech_coverage = {}
        for opp in session_data.get('opportunities', []):
            for tech in opp['purpose'].get('technologies', []):
                tech_coverage[tech] = tech_coverage.get(tech, 0) + 1
        
        # Problem type distribution  
        problem_types = {}
        for opp in session_data.get('opportunities', []):
            problem_type = opp['purpose'].get('primary_purpose', 'unknown')
            problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
        
        return SessionAnalytics(
            session_id=session_data['session_id'],
            start_time=start_time,
            end_time=end_time,
            total_opportunities=len(session_data.get('opportunities', [])),
            total_searches=len(session_data.get('actions', [])),
            total_repositories_analyzed=len([a for a in session_data.get('actions', []) if a.get('action_type') == 'repository_analysis']),
            total_emails_generated=len(session_data.get('emails', [])),
            average_confidence=sum(opp['purpose'].get('confidence_score', 0) for opp in session_data.get('opportunities', [])) / max(len(session_data.get('opportunities', [])), 1),
            average_success_score=performance_metrics.purpose_accuracy,
            technology_coverage=tech_coverage,
            problem_type_distribution=problem_types,
            quality_metrics={
                'search_efficiency': performance_metrics.search_efficiency,
                'solution_relevance': performance_metrics.solution_relevance,
                'analysis_depth': performance_metrics.analysis_depth_score
            }
        )
    
    def generate_recommendations(self, performance_metrics: PerformanceMetrics, learning_patterns: List[LearningPattern], quality_insights: List[QualityInsight]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if performance_metrics.search_efficiency < 0.6:
            recommendations.append("🔍 Improve search query specificity by including more technical terms and use cases")
        
        if performance_metrics.technology_match_rate < 0.5:
            recommendations.append("🔧 Enhance technology detection algorithms to better identify relevant tech stacks")
        
        if performance_metrics.analysis_depth_score < 0.7:
            recommendations.append("📊 Increase repository analysis depth for complex technical scenarios")
        
        # Pattern-based recommendations
        for pattern in learning_patterns:
            if pattern.pattern_type == "technology_preference" and pattern.confidence > 0.7:
                recommendations.append(f"⭐ Focus on {pattern.recommendation}")
        
        # Quality-based recommendations
        high_impact_insights = [insight for insight in quality_insights if insight.impact_level == "high"]
        for insight in high_impact_insights:
            recommendations.append(f"💡 {insight.suggested_action}")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def save_analytics_report(self, report: Dict[str, Any]) -> str:
        """Save analytics report to file"""
        
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"rag_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, cls=DateTimeEncoder)
            
        return str(report_file)
    
    def display_analytics_summary(self, report: Dict[str, Any]) -> None:
        """Display analytics summary to console"""
        
        print(f"\n📈 RAG ANALYTICS SUMMARY:")
        print(f"   📊 Session: {report['session_analytics']['session_id']}")
        print(f"   ⏱️  Duration: {report['session_analytics']['total_opportunities']} opportunities processed")
        print(f"   🎯 Avg Confidence: {report['session_analytics']['average_confidence']:.2f}")
        print(f"   ✅ Success Rate: {report['session_analytics']['average_success_score']:.2f}")
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        perf = report['performance_metrics']
        print(f"   🔍 Search Efficiency: {perf['search_efficiency']:.2f}")
        print(f"   📊 Analysis Depth: {perf['analysis_depth_score']:.2f}")
        print(f"   💡 Solution Relevance: {perf['solution_relevance']:.2f}")
        print(f"   🔧 Tech Match Rate: {perf['technology_match_rate']:.2f}")
        
        if report['recommendations']:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'][:3], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n📋 Full report saved to reports/")
