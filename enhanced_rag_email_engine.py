"""Enhanced RAG email generation engine with comprehensive transparency and analytics."""

import json
import logging
import os
import requests
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import ProcessedContent, EmailSolution, GitHubDiscoveryAction
from config import Config
from llm_processor import LLMProcessor
from llm_integration import RAGLLMIntegration
from enhanced_purpose_engine import TransparentRAGPurposeEngine, EnhancedRAGPurpose
from llm_search_strategist import LLMSearchStrategist, LLMSearchStrategy, RepositoryAnalysisStrategy
from enhanced_action_transparency import LLMDrivenActionTracker, EnhancedTransparentCommunicator, SearchExecutionResult, RepositoryAnalysisResult
from llm_outcome_assessor import LLMOutcomeAssessor, OutcomeAssessment
from enhanced_analytics import RAGAnalyticsEngine


class EnhancedRAGEmailEngine:
    """Enhanced RAG engine with comprehensive transparency and analytics"""
    
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_processor = LLMProcessor()
        self.llm_integration = RAGLLMIntegration(self.llm_processor)
        
        # Initialize enhanced LLM-driven components
        self.purpose_engine = TransparentRAGPurposeEngine(self.llm_integration)
        self.search_strategist = LLMSearchStrategist(self.llm_integration)
        self.action_tracker = LLMDrivenActionTracker()
        self.outcome_assessor = LLMOutcomeAssessor(self.llm_integration)
        self.analytics_engine = RAGAnalyticsEngine()
        self.communicator = EnhancedTransparentCommunicator()
        
        # Test LLM integration on initialization
        if self.llm_integration.test_llm_integration():
            self.logger.info("Enhanced LLM integration test passed")
        else:
            self.logger.warning("Enhanced LLM integration test failed - will use fallback methods")
        
        # Ensure directories exist
        os.makedirs("emails", exist_ok=True)
        os.makedirs("emails/logs", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
    
    def generate_rag_email_solutions(self, opportunities: List[ProcessedContent]) -> None:
        """Generate enhanced RAG email solutions with maximum LLM decision-making"""
        
        print(f"\n🚀 LLM-DRIVEN RAG EMAIL GENERATION STARTING...")
        print(f"   📊 Processing {len(opportunities)} opportunities with maximum LLM intelligence")
        
        # Initialize session analytics
        session_data = self.analytics_engine.session_logger.session_data
        
        for i, opportunity in enumerate(opportunities):
            print(f"\n📧 Processing Opportunity {i+1}/{len(opportunities)}")
            
            try:
                # Step 1: Enhanced Purpose Detection with LLM
                purpose = self.determine_enhanced_purpose(opportunity, i)
                self.analytics_engine.session_logger.log_opportunity_processing(i, purpose)
                
                # Step 2: LLM-Driven Repository Discovery
                github_actions = self.discover_with_llm_strategy(opportunity, purpose)
                
                # Step 3: Enhanced Email Generation
                email_solution = self.generate_enhanced_email_solution(opportunity, github_actions, purpose)
                self.analytics_engine.session_logger.log_email_generation(i, email_solution)
                
                # Step 4: Save Email Solution
                self.save_email_solution(email_solution, i+1)
                
            except Exception as e:
                self.logger.error(f"Error processing opportunity {i+1}: {e}")
                print(f"   ❌ Error: {e}")
        
        # Generate comprehensive analytics report
        self.analytics_engine.session_logger.finalize_session()
        comprehensive_report = self.analytics_engine.generate_comprehensive_report(session_data, self.action_tracker)
        
        # Generate final session analytics from LLM-driven components
        llm_analytics = self.action_tracker.generate_session_analytics()
        
        print(f"\n✅ LLM-DRIVEN RAG EMAIL GENERATION COMPLETE!")
        print(f"   📊 Generated {len(opportunities)} email solutions")
        print(f"   🧠 LLM Decisions Made: {llm_analytics['session_summary']['llm_decisions_made']}")
        print(f"   🔧 API Requests Used: {llm_analytics['session_summary']['total_api_requests_used']}/{llm_analytics['session_summary']['max_requests_allowed']}")
        print(f"   📈 Average LLM Confidence: {llm_analytics['session_summary']['average_confidence']:.2f}")
    
    def discover_with_llm_strategy(self, opportunity: ProcessedContent, purpose: EnhancedRAGPurpose) -> List[GitHubDiscoveryAction]:
        """LLM-driven repository discovery with adaptive strategy"""
        
        print(f"\n🧠 LLM-DRIVEN REPOSITORY DISCOVERY")
        
        # Check initial API availability
        self.action_tracker.communicator.announce_rate_limit_status(self.action_tracker.rate_limit_tracker)
        
        if not self.action_tracker.can_proceed_with_api_usage(3):
            self.logger.warning("Insufficient API requests for discovery - using fallback")
            return self._fallback_discovery(opportunity, purpose)
        
        # Step 1: Generate LLM search strategy
        search_strategy = self.search_strategist.generate_search_strategy(
            opportunity, purpose.reasoning
        )
        
        # Record LLM decision
        self.action_tracker.record_llm_decision(
            'search_strategy',
            {'opportunity': opportunity.original.title, 'purpose': purpose.primary_purpose},
            {'search_queries': search_strategy.search_queries, 'reasoning': search_strategy.reasoning},
            search_strategy.reasoning,
            search_strategy.confidence_score,
            search_strategy.estimated_api_usage
        )
        
        # Step 2: Execute search strategy
        search_result = self._execute_search_strategy(search_strategy)
        
        # Step 3: LLM assessment of search results
        search_assessment = self.outcome_assessor.assess_search_outcome(
            search_strategy, search_result, purpose.primary_purpose
        )
        
        # Step 4: Generate repository analysis strategy
        if search_result.repositories_found and self.action_tracker.can_proceed_with_api_usage(2):
            repo_strategy = self.search_strategist.generate_repository_analysis_strategy(
                search_result.repositories_found, purpose
            )
            
            # Record repository analysis decision
            self.action_tracker.record_llm_decision(
                'repository_analysis',
                {'repositories': [r.get('full_name', '') for r in search_result.repositories_found[:5]]},
                {'target_repositories': repo_strategy.target_repositories, 'reasoning': repo_strategy.reasoning},
                repo_strategy.reasoning,
                0.8,  # Default confidence for repo analysis
                repo_strategy.estimated_api_usage
            )
            
            # Step 5: Execute repository analysis
            analysis_result = self._execute_repository_analysis(repo_strategy, search_result.repositories_found)
            
            # Step 6: LLM assessment of repository analysis
            repo_assessment = self.outcome_assessor.assess_repository_analysis_outcome(
                repo_strategy, analysis_result, purpose.primary_purpose
            )
            
            # Generate final GitHub actions from analysis
            github_actions = self._convert_to_github_actions(analysis_result, search_result.repositories_found)
            
        else:
            # Convert search results directly to GitHub actions
            github_actions = self._convert_search_to_github_actions(search_result.repositories_found, purpose)
        
        # Step 7: Generate adaptive recommendations for future improvements
        if self.action_tracker.can_proceed_with_api_usage(1):
            remaining_calls = self.action_tracker.rate_limit_tracker.get_remaining_for_session()
            recommendations = self.outcome_assessor.generate_adaptive_recommendations(
                search_result.repositories_found, remaining_calls, purpose.primary_purpose
            )
            
            print(f"\n🎯 LLM ADAPTIVE RECOMMENDATIONS:")
            if recommendations.get('should_continue'):
                print(f"   ✅ Recommended: Continue with {len(recommendations.get('recommended_actions', []))} actions")
            else:
                print(f"   🛑 Recommended: Stop and proceed with current results")
        
        return github_actions
    
    def determine_enhanced_purpose(self, opportunity: ProcessedContent, opportunity_index: int) -> EnhancedRAGPurpose:
        """Enhanced purpose detection with transparency and reasoning"""
        
        print(f"\n🎯 ENHANCED PURPOSE DETECTION (Opportunity {opportunity_index+1})")
        
        # Use enhanced purpose engine
        enhanced_purpose = self.purpose_engine.generate_purpose_with_reasoning(opportunity)
        
        # Announce the decision with transparency
        self.communicator.announce_purpose_decision(enhanced_purpose)
        
        return enhanced_purpose
    
    def _execute_search_strategy(self, strategy: LLMSearchStrategy) -> SearchExecutionResult:
        """Execute the LLM-generated search strategy"""
        
        start_time = time.time()
        all_repositories = []
        queries_executed = []
        api_requests_used = 0
        
        print(f"\n🔍 EXECUTING LLM SEARCH STRATEGY:")
        print(f"   📋 Planned Queries: {len(strategy.search_queries)}")
        print(f"   🎯 Expected Results: {strategy.expected_outcomes.get('total_repositories', 'unknown')}")
        
        for query_info in strategy.search_queries:
            if not self.action_tracker.can_proceed_with_api_usage(1):
                print(f"   ⚠️  Stopping search execution - API limit reached")
                break
            
            query = query_info.get('query', '')
            filters = query_info.get('filters', {})
            language = filters.get('language')
            
            print(f"   🔍 Executing: '{query}' (Priority: {query_info.get('priority', 'medium')})")
            
            # Execute the search
            repos = self.search_github_repositories(query, language)
            api_requests_used += 1
            
            queries_executed.append({
                'query': query,
                'filters': filters,
                'results_count': len(repos),
                'reasoning': query_info.get('reasoning', '')
            })
            
            all_repositories.extend(repos)
            
            # Track API usage
            self.action_tracker.track_api_usage(1)
            
            print(f"      ✅ Found {len(repos)} repositories")
        
        execution_time = time.time() - start_time
        
        # Remove duplicates while preserving order
        unique_repos = []
        seen_names = set()
        for repo in all_repositories:
            repo_name = repo.get('full_name', '')
            if repo_name not in seen_names:
                unique_repos.append(repo)
                seen_names.add(repo_name)
        
        # Calculate success metrics
        expected_repos = strategy.expected_outcomes.get('total_repositories', 10)
        actual_repos = len(unique_repos)
        success_score = min(actual_repos / max(expected_repos, 1), 1.0)
        
        success_metrics = {
            'expected': expected_repos,
            'actual': actual_repos,
            'success_score': success_score,
            'target_met': actual_repos >= expected_repos,
            'repositories_found': actual_repos
        }
        
        # Identify gaps
        gaps_identified = []
        if actual_repos < expected_repos:
            gaps_identified.append(f"Found {actual_repos} repositories, expected {expected_repos}")
        
        if not any(repo.get('stargazers_count', 0) > 100 for repo in unique_repos):
            gaps_identified.append("No high-quality repositories (>100 stars) found")
        
        result = SearchExecutionResult(
            strategy=strategy,
            queries_executed=queries_executed,
            repositories_found=unique_repos,
            api_requests_used=api_requests_used,
            execution_time=execution_time,
            success_metrics=success_metrics,
            gaps_identified=gaps_identified
        )
        
        # Record the execution result
        self.action_tracker.record_search_execution(result)
        
        return result
    
    def _execute_repository_analysis(self, strategy: RepositoryAnalysisStrategy, available_repos: List[Dict]) -> RepositoryAnalysisResult:
        """Execute the LLM-generated repository analysis strategy"""
        
        print(f"\n📂 EXECUTING LLM REPOSITORY ANALYSIS:")
        print(f"   📦 Target Repositories: {len(strategy.target_repositories)}")
        print(f"   🔍 Analysis Depth: {strategy.analysis_depth}")
        
        repositories_analyzed = []
        files_examined = []
        code_patterns_found = []
        insights_extracted = []
        api_requests_used = 0
        
        # Create mapping of repo names to repo data
        repo_mapping = {repo.get('full_name', ''): repo for repo in available_repos}
        
        for repo_name in strategy.target_repositories:
            if not self.action_tracker.can_proceed_with_api_usage(1):
                print(f"   ⚠️  Stopping analysis - API limit reached")
                break
            
            if repo_name not in repo_mapping:
                print(f"   ❌ Repository {repo_name} not found in search results")
                continue
            
            repo_data = repo_mapping[repo_name]
            print(f"   📂 Analyzing: {repo_name}")
            
            # Analyze repository based on strategy
            files, snippets = self.analyze_repository_enhanced(repo_data, strategy)
            api_requests_used += 1
            
            repositories_analyzed.append(repo_name)
            files_examined.extend(files)
            
            # Extract patterns based on strategy
            patterns = self._extract_code_patterns(snippets, strategy.analysis_patterns)
            code_patterns_found.extend(patterns)
            
            # Generate insights
            insights = self._generate_insights(repo_data, files, patterns, strategy)
            insights_extracted.extend(insights)
            
            self.action_tracker.track_api_usage(1)
            
            print(f"      ✅ Files: {len(files)}, Patterns: {len(patterns)}, Insights: {len(insights)}")
        
        # Calculate analysis quality
        quality_score = self._calculate_analysis_quality(
            repositories_analyzed, files_examined, code_patterns_found, insights_extracted, strategy
        )
        
        result = RepositoryAnalysisResult(
            strategy=strategy,
            repositories_analyzed=repositories_analyzed,
            files_examined=files_examined,
            code_patterns_found=code_patterns_found,
            api_requests_used=api_requests_used,
            analysis_quality=quality_score,
            insights_extracted=insights_extracted
        )
        
        # Record the analysis result
        self.action_tracker.record_repository_analysis(result)
        
        return result
    
    def analyze_repository_enhanced(self, repo: Dict, strategy_or_purpose) -> tuple[List[str], List[str]]:
        """Enhanced repository analysis with pattern detection"""
        
        files_analyzed = []
        code_snippets = []
        
        try:
            # Get repository contents
            contents_url = f"https://api.github.com/repos/{repo['full_name']}/contents"
            response = requests.get(contents_url, headers=self.headers)
            
            if response.status_code == 200:
                contents = response.json()
                
                # Determine priority patterns based on input type
                if hasattr(strategy_or_purpose, 'expected_file_patterns'):
                    # It's a strategy with LLM-generated patterns
                    priority_patterns = strategy_or_purpose.expected_file_patterns
                elif hasattr(strategy_or_purpose, 'priority_files'):
                    # It's a strategy with priority files
                    priority_patterns = strategy_or_purpose.priority_files
                elif hasattr(strategy_or_purpose, 'expected_file_patterns'):
                    # It's a purpose
                    priority_patterns = strategy_or_purpose.expected_file_patterns
                else:
                    # Fallback
                    priority_patterns = ['auth', 'login', 'token', 'jwt', 'oauth', 'security']
                
                print(f"      🔍 Looking for patterns: {', '.join(priority_patterns[:5])}")
                
                for item in contents:
                    if item['type'] == 'file':
                        filename = item['name'].lower()
                        
                        # Check if file matches priority patterns
                        if any(pattern.lower() in filename for pattern in priority_patterns):
                            files_analyzed.append(item['name'])
                            
                            # Get file content for analysis
                            if len(code_snippets) < 3:  # Limit to 3 snippets
                                snippet = self.get_file_snippet(repo['full_name'], item['path'])
                                if snippet:
                                    code_snippets.append(snippet)
                        
                        # Always check README files
                        elif 'readme' in filename:
                            files_analyzed.append(item['name'])
                            if len(code_snippets) < 5:
                                snippet = self.get_file_snippet(repo['full_name'], item['path'])
                                if snippet:
                                    code_snippets.append(snippet)
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Repository analysis failed for {repo['full_name']}: {e}")
        
        return files_analyzed, code_snippets
    
    def get_file_snippet(self, repo_name: str, file_path: str) -> Optional[str]:
        """Get a snippet of file content from repository"""
        
        try:
            file_url = f"https://api.github.com/repos/{repo_name}/contents/{file_path}"
            response = requests.get(file_url, headers=self.headers)
            
            if response.status_code == 200:
                file_data = response.json()
                
                # Handle base64 encoded content
                if file_data.get('encoding') == 'base64':
                    import base64
                    content = base64.b64decode(file_data['content']).decode('utf-8', errors='ignore')
                    # Return first 500 characters
                    return content[:500]
                    
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get file snippet for {repo_name}/{file_path}: {e}")
            return None
    
    def _extract_code_patterns(self, snippets: List[str], analysis_patterns: List[str]) -> List[str]:
        """Extract code patterns from snippets based on analysis strategy"""
        
        patterns_found = []
        
        for snippet in snippets:
            snippet_lower = snippet.lower()
            
            for pattern in analysis_patterns:
                pattern_lower = pattern.lower()
                
                # Simple pattern matching - could be enhanced with LLM
                if any(keyword in snippet_lower for keyword in pattern_lower.split()):
                    patterns_found.append(f"Found {pattern} in code snippet")
        
        return patterns_found
    
    def _generate_insights(self, repo_data: Dict, files: List[str], patterns: List[str], strategy: RepositoryAnalysisStrategy) -> List[str]:
        """Generate insights from repository analysis"""
        
        insights = []
        
        # Repository quality insights
        stars = repo_data.get('stargazers_count', 0)
        if stars > 1000:
            insights.append(f"High-quality repository with {stars} stars indicates well-maintained code")
        
        # File analysis insights
        if any('test' in f.lower() for f in files):
            insights.append("Repository includes test files, suggesting good code quality")
        
        if any('docker' in f.lower() for f in files):
            insights.append("Repository includes Docker configuration for easy deployment")
        
        # Pattern insights
        if len(patterns) > 2:
            insights.append(f"Rich implementation with {len(patterns)} authentication patterns found")
        
        return insights
    
    def _calculate_analysis_quality(self, repos: List[str], files: List[str], patterns: List[str], insights: List[str], strategy: RepositoryAnalysisStrategy) -> float:
        """Calculate quality score for repository analysis"""
        
        quality = 0.0
        
        # Repository coverage
        target_repos = len(strategy.target_repositories)
        analyzed_repos = len(repos)
        repo_coverage = analyzed_repos / max(target_repos, 1)
        quality += repo_coverage * 0.3
        
        # File analysis coverage
        if len(files) >= 3:
            quality += 0.3
        elif len(files) >= 1:
            quality += 0.15
        
        # Pattern extraction success
        if len(patterns) >= 2:
            quality += 0.2
        elif len(patterns) >= 1:
            quality += 0.1
        
        # Insight generation
        if len(insights) >= 2:
            quality += 0.2
        elif len(insights) >= 1:
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _convert_to_github_actions(self, analysis_result: RepositoryAnalysisResult, repo_data: List[Dict]) -> List[GitHubDiscoveryAction]:
        """Convert repository analysis results to GitHub actions"""
        
        github_actions = []
        repo_mapping = {repo.get('full_name', ''): repo for repo in repo_data}
        
        for repo_name in analysis_result.repositories_analyzed:
            if repo_name in repo_mapping:
                repo = repo_mapping[repo_name]
                
                # Calculate relevance score based on analysis results
                relevance_score = self._calculate_relevance_from_analysis(repo_name, analysis_result)
                
                github_action = GitHubDiscoveryAction(
                    repository_name=repo_name,
                    purpose=f"LLM-driven analysis for authentication patterns",
                    relevance_score=relevance_score,
                    files_analyzed=[f for f in analysis_result.files_examined if repo_name in f or True],  # Simplified
                    code_snippets_found=len([p for p in analysis_result.code_patterns_found if repo_name in p]),
                    repository_stats={
                        'stars': repo.get('stargazers_count', 0),
                        'forks': repo.get('forks_count', 0),
                        'language': repo.get('language', 'Unknown'),
                        'updated': repo.get('updated_at', 'Unknown')
                    },
                    analysis_summary=f"LLM analysis found {len([i for i in analysis_result.insights_extracted if repo_name in i])} insights"
                )
                
                github_actions.append(github_action)
        
        return github_actions
    
    def _convert_search_to_github_actions(self, repositories: List[Dict], purpose: EnhancedRAGPurpose) -> List[GitHubDiscoveryAction]:
        """Convert search results directly to GitHub actions when analysis is skipped"""
        
        github_actions = []
        
        for repo in repositories[:5]:  # Top 5 repositories
            relevance_score = self.calculate_enhanced_relevance(repo, purpose)
            
            github_action = GitHubDiscoveryAction(
                repository_name=repo.get('full_name', 'unknown'),
                purpose=f"Search result for {purpose.primary_purpose}",
                relevance_score=relevance_score,
                files_analyzed=[],
                code_snippets_found=0,
                repository_stats={
                    'stars': repo.get('stargazers_count', 0),
                    'forks': repo.get('forks_count', 0),
                    'language': repo.get('language', 'Unknown'),
                    'updated': repo.get('updated_at', 'Unknown')
                },
                analysis_summary="Search result - analysis skipped due to API limits"
            )
            
            github_actions.append(github_action)
        
        return github_actions
    
    def _calculate_relevance_from_analysis(self, repo_name: str, analysis_result: RepositoryAnalysisResult) -> float:
        """Calculate relevance score based on analysis results"""
        
        relevance = 0.5  # Base score
        
        # Boost for patterns found
        patterns_for_repo = [p for p in analysis_result.code_patterns_found if repo_name in p]
        relevance += min(len(patterns_for_repo) * 0.1, 0.3)
        
        # Boost for insights
        insights_for_repo = [i for i in analysis_result.insights_extracted if repo_name in i]
        relevance += min(len(insights_for_repo) * 0.1, 0.2)
        
        return min(relevance, 1.0)
    
    def _fallback_discovery(self, opportunity: ProcessedContent, purpose: EnhancedRAGPurpose) -> List[GitHubDiscoveryAction]:
        """Fallback discovery when API limits are reached"""
        
        print(f"   ⚠️  Using fallback discovery due to API limits")
        
        # Simple search with primary technology
        primary_tech = purpose.technologies[0] if purpose.technologies else 'javascript'
        repos = self.search_github_repositories(f"{primary_tech} authentication", primary_tech)
        
        return self._convert_search_to_github_actions(repos[:3], purpose)
    
    def generate_enhanced_email_solution(self, opportunity: ProcessedContent, github_actions: List[GitHubDiscoveryAction], purpose: EnhancedRAGPurpose) -> EmailSolution:
        """Generate enhanced email solution with improved context"""
        
        print(f"\n📧 GENERATING ENHANCED EMAIL SOLUTION")
        
        # Build comprehensive GitHub context
        github_context = self.build_enhanced_github_context(github_actions, purpose)
        
        # Original user query
        original_query = f"{opportunity.original.title}\n{opportunity.original.content}"
        
        # Generate enhanced email content
        email_content = self.generate_enhanced_email_content(original_query, github_context, purpose, github_actions)
        
        # Calculate enhanced confidence score
        confidence_score = self.calculate_enhanced_confidence(github_actions, purpose, opportunity)
        
        return EmailSolution(
            original_query=original_query,
            email_content=email_content,
            github_actions=github_actions,
            confidence_score=confidence_score,
            solution_quality=self.assess_solution_quality(github_actions, purpose),
            generated_timestamp=datetime.now().isoformat(),
            purpose_reasoning=purpose.reasoning,
            success_metrics={
                'repositories_found': len(github_actions),
                'average_relevance': sum(action.relevance_score for action in github_actions) / len(github_actions) if github_actions else 0,
                'technology_coverage': len(purpose.technologies),
                'complexity_addressed': purpose.reasoning.technical_complexity if purpose.reasoning else 0
            }
        )
    
    def build_enhanced_purpose_queries(self, purpose: EnhancedRAGPurpose) -> List[str]:
        """Build enhanced search queries based on purpose analysis"""
        
        queries = []
        
        # Primary technology-focused queries
        for tech in purpose.technologies[:3]:  # Top 3 technologies
            queries.append(f"{tech} authentication")
            if purpose.reasoning and 'jwt' in purpose.reasoning.solution_requirements:
                queries.append(f"{tech} jwt authentication")
            if purpose.reasoning and 'oauth' in purpose.reasoning.solution_requirements:
                queries.append(f"{tech} oauth implementation")
        
        # Problem-type specific queries
        if purpose.reasoning:
            if purpose.reasoning.problem_type == 'debugging_issue':
                queries.append(f"authentication error handling {purpose.technologies[0] if purpose.technologies else 'javascript'}")
            elif purpose.reasoning.problem_type == 'implementation_guidance':
                queries.append(f"authentication tutorial {purpose.technologies[0] if purpose.technologies else 'javascript'}")
            elif purpose.reasoning.problem_type == 'scaling_challenge':
                queries.append(f"authentication scalable production {purpose.technologies[0] if purpose.technologies else 'javascript'}")
        
        # General authentication queries
        queries.extend([
            "authentication best practices",
            "user login implementation",
            "token management"
        ])
        
        return queries[:5]  # Top 5 queries
    
    def search_github_repositories(self, query: str, language: str = None) -> List[Dict]:
        """Search GitHub repositories with enhanced filtering"""
        
        url = f"https://api.github.com/search/repositories"
        
        # Build search query
        search_query = f"{query} in:name,description,readme"
        if language:
            search_query += f" language:{language}"
        
        # Add quality filters
        search_query += " stars:>5 archived:false"
        
        params = {
            'q': search_query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': 10
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('items', [])
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"GitHub search failed: {e}")
            return []
    
    def analyze_repository_enhanced(self, repo: Dict, purpose: EnhancedRAGPurpose) -> tuple[List[str], List[str]]:
        """Enhanced repository analysis with pattern detection"""
        
        files_analyzed = []
        code_snippets = []
        
        try:
            # Get repository contents
            contents_url = f"https://api.github.com/repos/{repo['full_name']}/contents"
            response = requests.get(contents_url, headers=self.headers)
            
            if response.status_code == 200:
                contents = response.json()
                
                # Priority files based on purpose
                priority_patterns = purpose.expected_file_patterns if purpose.expected_file_patterns else [
                    'auth', 'login', 'token', 'jwt', 'oauth', 'security'
                ]
                
                for item in contents:
                    if item['type'] == 'file':
                        filename = item['name'].lower()
                        
                        # Check if file matches priority patterns
                        if any(pattern.lower() in filename for pattern in priority_patterns):
                            files_analyzed.append(item['name'])
                            
                            # Get file content for analysis
                            if len(code_snippets) < 3:  # Limit to 3 snippets
                                snippet = self.get_file_snippet(repo['full_name'], item['path'])
                                if snippet:
                                    code_snippets.append(snippet)
                        
                        # Always check README files
                        elif 'readme' in filename:
                            files_analyzed.append(item['name'])
                            if len(code_snippets) < 3:
                                snippet = self.get_file_snippet(repo['full_name'], item['path'])
                                if snippet:
                                    code_snippets.append(snippet)
                        
                        # Stop if we have enough files
                        if len(files_analyzed) >= 5:
                            break
                            
        except Exception as e:
            self.logger.error(f"Error analyzing repository {repo['full_name']}: {e}")
        
        return files_analyzed, code_snippets
    
    def get_file_snippet(self, repo_name: str, file_path: str, max_lines: int = 50) -> Optional[str]:
        """Get a snippet of file content"""
        
        try:
            url = f"https://api.github.com/repos/{repo_name}/contents/{file_path}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                file_data = response.json()
                if file_data.get('encoding') == 'base64':
                    import base64
                    content = base64.b64decode(file_data['content']).decode('utf-8', errors='ignore')
                    
                    # Return first max_lines lines
                    lines = content.split('\n')[:max_lines]
                    return '\n'.join(lines)
                    
        except Exception as e:
            self.logger.error(f"Error getting file snippet: {e}")
        
        return None
    
    def calculate_enhanced_relevance(self, repo: Dict, purpose: EnhancedRAGPurpose) -> float:
        """Calculate enhanced relevance score"""
        
        score = 0.0
        
        # Technology match score
        if purpose.technologies:
            repo_text = f"{repo.get('name', '')} {repo.get('description', '')}".lower()
            tech_matches = sum(1 for tech in purpose.technologies if tech.lower() in repo_text)
            score += (tech_matches / len(purpose.technologies)) * 0.4
        
        # Stars score (normalized)
        stars = repo.get('stargazers_count', 0)
        star_score = min(stars / 100, 1.0) * 0.2
        score += star_score
        
        # Recency score
        updated_at = repo.get('updated_at', '')
        if updated_at:
            try:
                updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                days_since_update = (datetime.now().replace(tzinfo=None) - updated_date.replace(tzinfo=None)).days
                recency_score = max(0, (365 - days_since_update) / 365) * 0.2
                score += recency_score
            except:
                pass
        
        # Language match
        repo_language = repo.get('language', '').lower()
        if purpose.technologies and repo_language in [tech.lower() for tech in purpose.technologies]:
            score += 0.2
        
        return min(score, 1.0)
    
    def build_enhanced_github_context(self, github_actions: List[GitHubDiscoveryAction], purpose: EnhancedRAGPurpose) -> str:
        """Build enhanced GitHub context with structured information"""
        
        if not github_actions:
            return "No relevant GitHub repositories found for analysis."
        
        context_parts = []
        context_parts.append(f"## GitHub Repository Analysis for: {purpose.primary_purpose}")
        context_parts.append(f"**Technologies Focus**: {', '.join(purpose.technologies)}")
        context_parts.append(f"**Search Strategy**: {purpose.search_strategy}")
        
        if purpose.reasoning:
            context_parts.append(f"**Problem Type**: {purpose.reasoning.problem_type}")
            context_parts.append(f"**Technical Complexity**: {purpose.reasoning.technical_complexity}/10")
        
        context_parts.append("\n### Discovered Repositories:")
        
        for i, action in enumerate(github_actions, 1):
            context_parts.append(f"\n**{i}. {action.repository_name}**")
            context_parts.append(f"   - **Relevance Score**: {action.relevance_score:.2f}")
            context_parts.append(f"   - **Purpose**: {action.purpose}")
            context_parts.append(f"   - **Files Analyzed**: {', '.join(action.files_analyzed[:3])}")
            context_parts.append(f"   - **Code Patterns Found**: {action.code_snippets_found}")
            
            if hasattr(action, 'repository_stats'):
                stats = action.repository_stats
                context_parts.append(f"   - **Stats**: ⭐{stats.get('stars', 0)} | 🍴{stats.get('forks', 0)} | 🗣️{stats.get('language', 'Unknown')}")
            
            if hasattr(action, 'analysis_summary'):
                context_parts.append(f"   - **Analysis**: {action.analysis_summary}")
        
        return '\n'.join(context_parts)
    
    def generate_enhanced_email_content(self, original_query: str, github_context: str, purpose: EnhancedRAGPurpose, github_actions: List[GitHubDiscoveryAction]) -> str:
        """Generate enhanced email content with comprehensive context"""
        
        enhanced_prompt = f"""You are a senior technical consultant specializing in authentication solutions.

CLIENT'S ORIGINAL QUESTION:
{original_query}

TECHNICAL ANALYSIS:
- Problem Type: {purpose.reasoning.problem_type if purpose.reasoning else 'General consultation'}
- Technologies: {', '.join(purpose.technologies)}
- Complexity Level: {purpose.reasoning.technical_complexity if purpose.reasoning else 'Medium'}/10
- Business Context: {purpose.reasoning.business_context if purpose.reasoning else 'Not specified'}

GITHUB RESEARCH FINDINGS:
{github_context}

SOLUTION REQUIREMENTS:
{', '.join(purpose.reasoning.solution_requirements) if purpose.reasoning and purpose.reasoning.solution_requirements else 'Standard implementation guidance'}

Your task: Write a professional, comprehensive email response that:

1. **Acknowledges** the client's specific question and context
2. **Provides** a clear, actionable solution based on the GitHub research
3. **Includes** specific repository recommendations with implementation guidance
4. **Addresses** the technical complexity appropriate to their skill level
5. **Offers** next steps and additional resources

Email Structure:
- Professional greeting
- Problem acknowledgment
- Solution overview
- Specific repository recommendations (with brief descriptions)
- Implementation guidance
- Next steps
- Professional closing

Write in a consultative, expert tone. Be specific about implementation details while remaining accessible."""
        
        try:
            email_content = self.llm_integration.generate_email_with_llm(enhanced_prompt)
            
            if not email_content:
                # Enhanced fallback
                email_content = self.generate_enhanced_fallback_email(original_query, github_actions, purpose)
            
            return email_content
            
        except Exception as e:
            self.logger.error(f"Enhanced email generation failed: {e}")
            return self.generate_enhanced_fallback_email(original_query, github_actions, purpose)
    
    def generate_enhanced_fallback_email(self, original_query: str, github_actions: List[GitHubDiscoveryAction], purpose: EnhancedRAGPurpose) -> str:
        """Generate enhanced fallback email when LLM fails"""
        
        email_parts = []
        email_parts.append("Hello,")
        email_parts.append("")
        email_parts.append("Thank you for your authentication implementation question. Based on our technical analysis and GitHub research, here's a comprehensive solution:")
        email_parts.append("")
        
        # Problem acknowledgment
        if purpose.reasoning:
            email_parts.append(f"**Problem Analysis:**")
            email_parts.append(f"- Type: {purpose.reasoning.problem_type}")
            email_parts.append(f"- Technologies: {', '.join(purpose.technologies)}")
            email_parts.append(f"- Complexity: {purpose.reasoning.technical_complexity}/10")
            email_parts.append("")
        
        # Repository recommendations
        email_parts.append("**Recommended GitHub Repositories:**")
        email_parts.append("")
        
        for i, action in enumerate(github_actions[:3], 1):
            email_parts.append(f"{i}. **{action.repository_name}** (Relevance: {action.relevance_score:.1f}/1.0)")
            email_parts.append(f"   - {action.purpose}")
            email_parts.append(f"   - Found {action.code_snippets_found} relevant code patterns")
            if hasattr(action, 'repository_stats'):
                stats = action.repository_stats
                email_parts.append(f"   - {stats.get('stars', 0)} stars, {stats.get('language', 'Unknown')} language")
            email_parts.append("")
        
        # Implementation guidance
        email_parts.append("**Implementation Approach:**")
        if purpose.reasoning and purpose.reasoning.solution_requirements:
            for req in purpose.reasoning.solution_requirements[:3]:
                email_parts.append(f"- {req}")
        else:
            email_parts.append("- Review the recommended repositories for implementation patterns")
            email_parts.append("- Start with the highest-rated repository for your technology stack")
            email_parts.append("- Adapt the examples to your specific use case")
        
        email_parts.append("")
        email_parts.append("**Next Steps:**")
        email_parts.append("1. Review the recommended repositories")
        email_parts.append("2. Test implementations in a development environment")
        email_parts.append("3. Adapt solutions to your specific requirements")
        email_parts.append("")
        email_parts.append("Feel free to reach out if you need further clarification on any of these recommendations.")
        email_parts.append("")
        email_parts.append("Best regards,")
        email_parts.append("Technical Solutions Team")
        
        return '\n'.join(email_parts)
    
    def calculate_enhanced_confidence(self, github_actions: List[GitHubDiscoveryAction], purpose: EnhancedRAGPurpose, opportunity: ProcessedContent) -> float:
        """Calculate enhanced confidence score"""
        
        confidence = 0.5  # Base confidence
        
        # Repository quality score
        if github_actions:
            avg_relevance = sum(action.relevance_score for action in github_actions) / len(github_actions)
            confidence += avg_relevance * 0.3
        
        # Purpose detection confidence
        confidence += purpose.confidence_score * 0.2
        
        # Technology coverage
        if purpose.technologies:
            tech_coverage = min(len(purpose.technologies) / 3, 1.0)  # Up to 3 technologies ideal
            confidence += tech_coverage * 0.1
        
        # Code snippets found
        total_snippets = sum(action.code_snippets_found for action in github_actions)
        snippet_score = min(total_snippets / 5, 1.0)  # Up to 5 snippets ideal
        confidence += snippet_score * 0.1
        
        return min(confidence, 0.95)
    
    def assess_solution_quality(self, github_actions: List[GitHubDiscoveryAction], purpose: EnhancedRAGPurpose) -> str:
        """Assess solution quality"""
        
        if not github_actions:
            return "low_quality"
        
        avg_relevance = sum(action.relevance_score for action in github_actions) / len(github_actions)
        total_snippets = sum(action.code_snippets_found for action in github_actions)
        
        if avg_relevance > 0.7 and total_snippets >= 3:
            return "high_quality"
        elif avg_relevance > 0.5 and total_snippets >= 1:
            return "medium_quality"
        else:
            return "low_quality"
    
    def save_email_solution(self, email_solution: EmailSolution, email_number: int) -> None:
        """Save enhanced email solution with comprehensive metadata"""
        
        filename = f"emails/email{email_number}.json"
        
        # Enhanced email data with all metadata
        email_data = {
            "original_user_query": email_solution.original_query,
            "email_solution": email_solution.email_content,
            "metadata": {
                "generated_timestamp": email_solution.generated_timestamp,
                "confidence_score": email_solution.confidence_score,
                "solution_quality": email_solution.solution_quality,
                "github_actions_count": len(email_solution.github_actions),
                "success_metrics": email_solution.success_metrics if hasattr(email_solution, 'success_metrics') else {}
            },
            "github_actions": [asdict(action) for action in email_solution.github_actions],
            "purpose_reasoning": asdict(email_solution.purpose_reasoning) if hasattr(email_solution, 'purpose_reasoning') and email_solution.purpose_reasoning else {}
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(email_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Email solution saved: {filename}")
            self.logger.info(f"Enhanced email solution saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save email solution: {e}")
            print(f"   ❌ Failed to save email solution: {e}")


# Backward compatibility - use enhanced engine
RAGEmailEngine = EnhancedRAGEmailEngine
