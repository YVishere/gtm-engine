"""Enhanced RAG email generation engine with comprehensive transparency and analytics."""

import json
import logging
import os
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from models import ProcessedContent, EmailSolution, GitHubDiscoveryAction
from config import Config
from llm_processor import LLMProcessor
from llm_integration import RAGLLMIntegration
from enhanced_purpose_engine import TransparentRAGPurposeEngine, EnhancedRAGPurpose
from action_transparency import RAGActionTracker, TransparentCommunicator
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
        
        # Initialize enhanced components
        self.purpose_engine = TransparentRAGPurposeEngine(self.llm_integration)
        self.action_tracker = RAGActionTracker()
        self.analytics_engine = RAGAnalyticsEngine()
        self.communicator = TransparentCommunicator()
        
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
        """Generate enhanced RAG email solutions with full transparency"""
        
        print(f"\n🚀 ENHANCED RAG EMAIL GENERATION STARTING...")
        print(f"   📊 Processing {len(opportunities)} opportunities with full transparency")
        
        # Start analytics session
        session_data = self.analytics_engine.session_logger.session_data
        
        for i, opportunity in enumerate(opportunities):
            print(f"\n📧 Processing Opportunity {i+1}/{len(opportunities)}")
            
            try:
                # Step 1: Enhanced Purpose Detection with Transparency
                purpose = self.determine_enhanced_purpose(opportunity, i)
                self.analytics_engine.session_logger.log_opportunity_processing(i, purpose)
                
                # Step 2: Transparent Repository Discovery
                github_actions = self.discover_with_enhanced_purpose(opportunity, purpose)
                
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
        
        print(f"\n✅ ENHANCED RAG EMAIL GENERATION COMPLETE!")
        print(f"   📊 Generated {len(opportunities)} email solutions with full analytics")
    
    def determine_enhanced_purpose(self, opportunity: ProcessedContent, opportunity_index: int) -> EnhancedRAGPurpose:
        """Enhanced purpose detection with transparency and reasoning"""
        
        print(f"\n🎯 ENHANCED PURPOSE DETECTION (Opportunity {opportunity_index+1})")
        
        # Use enhanced purpose engine
        enhanced_purpose = self.purpose_engine.generate_purpose_with_reasoning(opportunity)
        
        # Announce the decision with transparency
        self.communicator.announce_purpose_decision(enhanced_purpose)
        
        return enhanced_purpose
    
    def discover_with_enhanced_purpose(self, opportunity: ProcessedContent, purpose: EnhancedRAGPurpose) -> List[GitHubDiscoveryAction]:
        """Enhanced repository discovery with action tracking"""
        
        print(f"\n🔍 ENHANCED REPOSITORY DISCOVERY")
        
        # Track search actions transparently
        search_queries = self.build_enhanced_purpose_queries(purpose)
        discovered_repos = []
        
        for query in search_queries:
            # Track search action
            search_action = self.action_tracker.track_search_action(
                query, 'repositories', purpose,
                language=purpose.technologies[0] if purpose.technologies else None,
                sort='stars',
                order='desc'
            )
            
            # Execute search
            repos = self.search_github_repositories(query, purpose.technologies[0] if purpose.technologies else None)
            
            # Assess search outcome
            outcome_assessment = self.action_tracker.assess_action_outcome(
                f"search_{len(self.action_tracker.search_actions)}",
                f"Find {purpose.success_criteria.get('minimum_repositories', 3)} relevant repositories",
                f"Found {len(repos)} repositories",
                {'found_repositories': len(repos), 'relevance_score': 0.7}
            )
            
            discovered_repos.extend(repos)
        
        # Analyze repositories with transparency
        github_actions = []
        for repo in discovered_repos[:5]:  # Top 5 repositories
            
            # Track repository analysis action
            repo_action = self.action_tracker.track_repository_action(
                repo['full_name'], 'auth_patterns', purpose,
                target_files=purpose.expected_file_patterns[:3]
            )
            
            # Analyze repository
            files_analyzed, code_snippets = self.analyze_repository_enhanced(repo, purpose)
            relevance_score = self.calculate_enhanced_relevance(repo, purpose)
            
            github_action = GitHubDiscoveryAction(
                repository_name=repo['full_name'],
                purpose=f"Analyze for {purpose.primary_purpose}",
                relevance_score=relevance_score,
                files_analyzed=files_analyzed,
                code_snippets_found=len(code_snippets),
                repository_stats={
                    'stars': repo.get('stargazers_count', 0),
                    'forks': repo.get('forks_count', 0),
                    'language': repo.get('language', 'Unknown'),
                    'updated': repo.get('updated_at', 'Unknown')
                },
                analysis_summary=f"Found {len(code_snippets)} relevant code patterns"
            )
            
            github_actions.append(github_action)
        
        # Announce discovery results
        self.communicator.announce_discovery_results(github_actions, purpose)
        
        return github_actions
    
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
