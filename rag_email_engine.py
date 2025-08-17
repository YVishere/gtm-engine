"""RAG-enhanced email generation engine with comprehensive transparency and analytics."""

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

class RAGEmailEngine:
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
            self.logger.info("LLM integration test passed")
        else:
            self.logger.warning("LLM integration test failed - will use fallback methods")
        
        # Ensure emails directory exists
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
    
    # Legacy method compatibility - now calls enhanced methods
    def determine_search_purpose(self, opportunity: ProcessedContent) -> EnhancedRAGPurpose:
        """RAG model intelligently determines search purpose"""
        
        question_text = f"{opportunity.original.title} {opportunity.original.content}"
        technologies = self.extract_technologies(question_text, opportunity.key_topics)
        
        purpose_prompt = f"""You are a technical research assistant determining GitHub search strategy.

TECHNICAL QUESTION:
{question_text}

IDENTIFIED TOPICS: {opportunity.key_topics}
URGENCY LEVEL: {opportunity.urgency_level}

Your task: Determine the optimal GitHub search strategy for finding relevant code solutions.

Analyze and decide:
1. PRIMARY_PURPOSE: What specific technical solution/implementation to find (be specific)
2. TECHNOLOGIES: Key technologies/frameworks involved (extract from question)
3. SEARCH_STRATEGY: How to search (examples, documentation, implementations, tutorials)
4. URGENCY_CONTEXT: Context for prioritization based on urgency level

Return ONLY valid JSON in this format:
{{
    "primary_purpose": "Specific technical goal (e.g., 'JWT refresh token implementation with error handling')",
    "technologies": ["tech1", "tech2", "tech3"],
    "search_strategy": "Search approach (e.g., 'Find working examples with production-ready error handling')",
    "urgency_context": "Priority context (e.g., 'High-priority production issue requiring immediate solution')"
}}"""

        try:
            # Use the proper LLM integration
            purpose_data = self.llm_integration.generate_purpose_with_llm(purpose_prompt)
            
            if purpose_data:
                self.logger.info("Successfully generated purpose via LLM")
                return RAGSearchPurpose(
                    primary_purpose=purpose_data.get('primary_purpose', f'Authentication solution for {technologies[0] if technologies else "web application"}'),
                    technologies=purpose_data.get('technologies', technologies),
                    search_strategy=purpose_data.get('search_strategy', 'Find working examples and implementations'),
                    urgency_context=purpose_data.get('urgency_context', f"{opportunity.urgency_level.title()} priority technical assistance")
                )
            else:
                self.logger.warning("LLM returned empty purpose data, using fallback")
                return self.fallback_search_purpose(opportunity, technologies)
                
        except Exception as e:
            self.logger.warning(f"Failed to determine search purpose via LLM: {e}")
            return self.fallback_search_purpose(opportunity, technologies)
    
    def extract_technologies(self, question_text: str, topics: List[str]) -> List[str]:
        """Extract technology stack from question text and topics"""
        
        tech_mapping = {
            'react': ['react', 'reactjs', 'jsx', 'hooks'],
            'vue': ['vue', 'vuejs', 'nuxt'],
            'angular': ['angular', 'ng', 'typescript'],
            'node': ['node', 'nodejs', 'express', 'npm'],
            'python': ['python', 'django', 'flask', 'fastapi', 'pip'],
            'java': ['java', 'spring', 'springboot', 'maven'],
            'php': ['php', 'laravel', 'symfony', 'composer'],
            'go': ['go', 'golang', 'gin', 'gorilla'],
            'rust': ['rust', 'actix', 'cargo'],
            'javascript': ['javascript', 'js', 'es6', 'babel'],
            'jwt': ['jwt', 'jsonwebtoken', 'token'],
            'oauth': ['oauth', 'oauth2', 'openid', 'oidc'],
            'saml': ['saml', 'sso'],
            'nextjs': ['nextjs', 'next.js', 'vercel'],
            'mongodb': ['mongodb', 'mongo', 'mongoose'],
            'redis': ['redis', 'cache'],
            'docker': ['docker', 'container']
        }
        
        text_lower = question_text.lower()
        detected_techs = []
        
        # Check direct technology mentions
        for tech, keywords in tech_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_techs.append(tech)
        
        # Add relevant topics
        for topic in topics:
            topic_clean = topic.lower().replace('-', '').replace('_', '')
            if topic_clean not in [t.lower() for t in detected_techs] and len(topic_clean) > 2:
                detected_techs.append(topic)
        
        return detected_techs[:4] if detected_techs else ['authentication']
    
    def parse_purpose_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for search purpose"""
        try:
            # Clean response
            if '</thinking>' in response:
                response = response.split('</thinking>')[-1].strip()
            
            # Extract JSON
            if '```json' in response:
                json_part = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                json_part = response.split('```')[1].strip()
            else:
                json_part = response.strip()
            
            # Find JSON object
            start = json_part.find('{')
            end = json_part.rfind('}') + 1
            if start >= 0 and end > start:
                json_part = json_part[start:end]
            
            return json.loads(json_part)
        except Exception as e:
            self.logger.warning(f"Failed to parse purpose response: {e}")
            return {}
    
    def fallback_search_purpose(self, opportunity: ProcessedContent, technologies: List[str]) -> RAGSearchPurpose:
        """Fallback search purpose when LLM fails"""
        
        primary_tech = technologies[0] if technologies else 'web'
        urgency_map = {
            'high': 'Critical production issue',
            'medium': 'Important development task', 
            'low': 'Development research'
        }
        
        return RAGSearchPurpose(
            primary_purpose=f"{primary_tech} authentication implementation",
            technologies=technologies,
            search_strategy="Find working examples and best practices",
            urgency_context=urgency_map.get(opportunity.urgency_level, 'Development assistance')
        )
    
    def discover_with_purpose(self, opportunity: ProcessedContent, purpose: RAGSearchPurpose) -> List[GitHubDiscoveryAction]:
        """Discover repositories with detailed action tracking"""
        
        actions = []
        
        # Build purpose-driven queries
        queries = self.build_purpose_queries(purpose)
        
        for query in queries:
            print(f"   🔍 Searching GitHub: {query}")
            try:
                repos = self.search_repositories(query)
                
                for repo in repos[:2]:  # Top 2 per query to avoid rate limits
                    repo_name = f"{repo['owner']['login']}/{repo['name']}"
                    
                    # Determine specific purpose for this repo
                    repo_purpose = f"Analyzing {repo_name} for {purpose.primary_purpose}"
                    
                    # Analyze repository contents
                    files_analyzed, code_snippets = self.analyze_repository(repo, purpose)
                    
                    action = GitHubDiscoveryAction(
                        repository_name=repo_name,
                        purpose=repo_purpose,
                        relevance_score=self.calculate_relevance(repo, purpose),
                        files_analyzed=files_analyzed,
                        code_snippets_found=len(code_snippets)
                    )
                    
                    actions.append(action)
                    print(f"   📂 {repo_name}: {len(files_analyzed)} files, {len(code_snippets)} snippets")
                    
            except Exception as e:
                self.logger.warning(f"Error searching query '{query}': {e}")
                continue
        
        return sorted(actions, key=lambda x: x.relevance_score, reverse=True)[:5]
    
    def build_purpose_queries(self, purpose: RAGSearchPurpose) -> List[str]:
        """Build GitHub search queries based on purpose"""
        
        queries = []
        techs = [tech for tech in purpose.technologies if len(tech) > 2]
        
        # Primary technology + auth
        if techs:
            main_tech = techs[0]
            queries.append(f"{main_tech} authentication examples stars:>10")
            queries.append(f"{main_tech} jwt oauth implementation")
        
        # Specific purpose-based queries
        if 'jwt' in purpose.primary_purpose.lower():
            queries.append("jwt refresh token implementation javascript")
            queries.append("jwt authentication middleware examples")
        
        if 'oauth' in purpose.primary_purpose.lower():
            queries.append("oauth2 implementation examples")
            queries.append("oauth callback handling best practices")
        
        # Multi-tech queries
        if len(techs) > 1:
            combined = ' '.join(techs[:2])
            queries.append(f"{combined} authentication integration")
        
        # Fallback generic query
        if not queries:
            queries.append("authentication implementation examples stars:>50")
        
        return queries[:4]  # Limit queries to avoid rate limits
    
    def search_repositories(self, query: str, max_repos: int = 5) -> List[Dict]:
        """Search GitHub repositories"""
        
        url = "https://api.github.com/search/repositories"
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': max_repos
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('items', [])
        except Exception as e:
            self.logger.error(f"GitHub API error: {e}")
            return []
    
    def analyze_repository(self, repo: Dict, purpose: RAGSearchPurpose) -> tuple[List[str], List[str]]:
        """Analyze repository contents for relevant files"""
        
        owner = repo['owner']['login']
        repo_name = repo['name']
        files_analyzed = []
        code_snippets = []
        
        try:
            # Get repository contents
            contents_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents"
            response = requests.get(contents_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                contents = response.json()
                
                # Priority file patterns based on purpose
                priority_patterns = self.get_priority_patterns(purpose)
                
                # Check files in root
                for item in contents:
                    if item['type'] == 'file':
                        relevance = self.calculate_file_relevance(item['name'], priority_patterns)
                        if relevance > 0:
                            files_analyzed.append(item['name'])
                            # Would extract code snippets here in full implementation
                            code_snippets.append(f"// Code from {item['name']}")
                
                # Check common directories (limited to avoid rate limits)
                common_dirs = ['examples', 'docs', 'src']
                for item in contents:
                    if item['type'] == 'dir' and item['name'] in common_dirs:
                        dir_files = self.explore_directory_limited(owner, repo_name, item['name'], priority_patterns)
                        files_analyzed.extend(dir_files)
                        code_snippets.extend([f"// Code from {f}" for f in dir_files])
                        
                        if len(files_analyzed) >= 5:  # Limit to avoid excessive API calls
                            break
        
        except Exception as e:
            self.logger.warning(f"Error analyzing repository {owner}/{repo_name}: {e}")
        
        return files_analyzed[:5], code_snippets[:3]  # Limit results
    
    def explore_directory_limited(self, owner: str, repo_name: str, dir_name: str, priority_patterns: List[str]) -> List[str]:
        """Explore directory with limited depth to avoid rate limits"""
        
        try:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{dir_name}"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                contents = response.json()
                relevant_files = []
                
                for item in contents[:10]:  # Limit to first 10 items
                    if item['type'] == 'file':
                        relevance = self.calculate_file_relevance(item['name'], priority_patterns)
                        if relevance > 0:
                            relevant_files.append(f"{dir_name}/{item['name']}")
                
                return relevant_files[:3]  # Max 3 files per directory
        except:
            pass
        
        return []
    
    def get_priority_patterns(self, purpose: RAGSearchPurpose) -> List[str]:
        """Get file patterns based on search purpose"""
        
        patterns = ['auth', 'login', 'jwt', 'oauth', 'token']
        
        for tech in purpose.technologies:
            if tech.lower() == 'react':
                patterns.extend(['component', 'hook', 'context'])
            elif tech.lower() in ['node', 'nodejs']:
                patterns.extend(['middleware', 'route', 'server'])
            elif tech.lower() == 'python':
                patterns.extend(['views', 'models', 'middleware'])
        
        if 'jwt' in purpose.primary_purpose.lower():
            patterns.extend(['refresh', 'token', 'verify'])
        
        if 'oauth' in purpose.primary_purpose.lower():
            patterns.extend(['callback', 'provider', 'oauth'])
        
        return list(set(patterns))
    
    def calculate_file_relevance(self, filename: str, priority_patterns: List[str]) -> float:
        """Calculate file relevance score"""
        
        score = 0
        filename_lower = filename.lower()
        
        # Boost for priority patterns
        for pattern in priority_patterns:
            if pattern in filename_lower:
                score += 2
        
        # File type bonuses
        if filename.endswith('.md') and any(word in filename_lower for word in ['readme', 'auth', 'jwt', 'oauth']):
            score += 3
        elif filename.endswith(('.js', '.ts', '.py', '.go', '.java')) and 'auth' in filename_lower:
            score += 2
        elif filename.endswith(('.js', '.ts', '.py', '.go', '.java')) and any(word in filename_lower for word in ['example', 'demo', 'test']):
            score += 1
        
        return score
    
    def calculate_relevance(self, repo: Dict, purpose: RAGSearchPurpose) -> float:
        """Calculate repository relevance score"""
        
        score = 0
        
        # Base score from GitHub metrics
        score += min(repo['stargazers_count'] / 1000, 5)  # Stars
        score += min(repo['forks_count'] / 100, 2)        # Forks
        
        # Technology match
        repo_text = f"{repo['name']} {repo['description'] or ''}".lower()
        for tech in purpose.technologies:
            if tech.lower() in repo_text:
                score += 2
        
        # Purpose match
        purpose_keywords = purpose.primary_purpose.lower().split()
        for keyword in purpose_keywords:
            if len(keyword) > 3 and keyword in repo_text:
                score += 1
        
        # Language bonus
        if repo.get('language'):
            lang = repo['language'].lower()
            if lang in [tech.lower() for tech in purpose.technologies]:
                score += 1
        
        # Recent activity bonus
        try:
            from datetime import datetime, timezone
            updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
            days_old = (datetime.now(timezone.utc) - updated).days
            if days_old < 365:  # Updated within a year
                score += 1
        except:
            pass
        
        return min(score, 10)  # Cap at 10
    
    def generate_email_solution(self, opportunity: ProcessedContent, github_actions: List[GitHubDiscoveryAction], purpose: RAGSearchPurpose) -> EmailSolution:
        """Generate complete email solution with original query"""
        
        # Extract original user query
        original_query = self.extract_user_query(opportunity)
        
        # Build GitHub context summary
        github_context = self.build_github_context(github_actions, purpose)
        
        # Generate email content
        email_content = self.generate_email_content(original_query, github_context, purpose, github_actions)
        
        return EmailSolution(
            original_user_query=original_query,
            email_content=email_content,
            github_actions=github_actions,
            confidence_score=self.calculate_confidence(github_actions),
            generated_timestamp=datetime.now().isoformat()
        )
    
    def extract_user_query(self, opportunity: ProcessedContent) -> str:
        """Extract the original user question from opportunity"""
        
        # Use the original title and content as the user query
        title = opportunity.original.title
        content_preview = opportunity.original.content[:300] if opportunity.original.content else ""
        
        if content_preview:
            return f"{title}\n\n{content_preview}..."
        else:
            return title
    
    def build_github_context(self, github_actions: List[GitHubDiscoveryAction], purpose: RAGSearchPurpose) -> str:
        """Build GitHub context summary for email"""
        
        if not github_actions:
            return "I searched through several authentication-related repositories but didn't find specific examples for this exact use case."
        
        context_parts = []
        context_parts.append(f"I analyzed {len(github_actions)} relevant repositories for {purpose.primary_purpose}:")
        context_parts.append("")
        
        for i, action in enumerate(github_actions[:3], 1):
            repo_name = action.repository_name
            files_count = len(action.files_analyzed)
            snippets_count = action.code_snippets_found
            
            context_parts.append(f"{i}. **{repo_name}** (⭐ Relevance: {action.relevance_score:.1f}/10)")
            context_parts.append(f"   - Analyzed {files_count} relevant files")
            context_parts.append(f"   - Found {snippets_count} useful code examples")
            context_parts.append(f"   - Purpose: {action.purpose}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_email_content(self, original_query: str, github_context: str, purpose: RAGSearchPurpose, github_actions: List[GitHubDiscoveryAction]) -> str:
        """Generate the complete email content"""
        
        # Determine subject based on technologies
        tech_focus = purpose.technologies[0] if purpose.technologies else "authentication"
        subject = f"Re: {purpose.primary_purpose.title()} - Solution with Code Examples"
        
        email_template = f"""ORIGINAL USER QUERY:
{original_query}

---

EMAIL SOLUTION:

Subject: {subject}

Hi there,

I saw your question about {purpose.primary_purpose}. This is a common challenge that many developers face, especially when working with {', '.join(purpose.technologies[:3])}.

{github_context}

Based on these implementations, here's what I recommend:

1. **Best Practice Approach**: Start with the patterns shown in the top-rated repositories
2. **Security Considerations**: Make sure to implement proper error handling and token validation
3. **Production Ready**: The examples I found include production-tested implementations

Key takeaways from the code analysis:
- Focus on secure token handling and refresh mechanisms
- Implement proper error boundaries and fallback strategies  
- Consider using established libraries rather than building from scratch

For enterprise-grade authentication that handles all these complexities automatically, you might want to check out Descope. It provides production-ready auth with minimal implementation effort, which could save you weeks of development time.

Feel free to reach out if you need clarification on any of the implementation details!

Best regards,
Technical Solutions Team

---

GITHUB REPOSITORIES ANALYZED:
{self.format_github_actions_summary(github_actions)}
"""
        
        return email_template
    
    def format_github_actions_summary(self, github_actions: List[GitHubDiscoveryAction]) -> str:
        """Format GitHub actions summary for email footer"""
        
        if not github_actions:
            return "No repositories analyzed"
        
        summary_lines = []
        for action in github_actions:
            summary_lines.append(f"• {action.repository_name} - {action.purpose} (Score: {action.relevance_score:.1f})")
        
        return "\n".join(summary_lines)
    
    def calculate_confidence(self, github_actions: List[GitHubDiscoveryAction]) -> float:
        """Calculate confidence score for the email solution"""
        
        if not github_actions:
            return 0.3
        
        # Base confidence from number of repos
        base_confidence = min(len(github_actions) * 0.15, 0.6)
        
        # Boost from relevance scores
        avg_relevance = sum(action.relevance_score for action in github_actions) / len(github_actions)
        relevance_boost = min(avg_relevance / 10, 0.3)
        
        # Boost from code snippets found
        total_snippets = sum(action.code_snippets_found for action in github_actions)
        snippet_boost = min(total_snippets * 0.05, 0.2)
        
        total_confidence = base_confidence + relevance_boost + snippet_boost
        return min(total_confidence, 0.95)  # Cap at 95%
    
    def save_email_solutions(self, email_solutions: List[EmailSolution]) -> None:
        """Save email solutions as email1.json, email2.json, email3.json"""
        
        # Save individual emails (overwrite existing)
        for i, solution in enumerate(email_solutions[:3], 1):  # Max 3 emails
            filename = f"emails/email{i}.json"
            
            # Convert to dict for JSON serialization
            solution_dict = {
                'original_user_query': solution.original_user_query,
                'email_content': solution.email_content,
                'github_actions': [asdict(action) for action in solution.github_actions],
                'confidence_score': solution.confidence_score,
                'generated_timestamp': solution.generated_timestamp
            }
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(solution_dict, f, indent=2, ensure_ascii=False)
                print(f"   💾 Saved {filename}")
            except Exception as e:
                self.logger.error(f"Failed to save {filename}: {e}")
        
        # Save metadata
        metadata = {
            'total_emails_generated': len(email_solutions),
            'generation_timestamp': datetime.now().isoformat(),
            'avg_confidence': sum(s.confidence_score for s in email_solutions) / len(email_solutions) if email_solutions else 0,
            'total_repos_analyzed': sum(len(s.github_actions) for s in email_solutions)
        }
        
        try:
            with open('emails/email_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save email metadata: {e}")
        
        # Log generation details
        log_filename = f"emails/logs/generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(f"RAG Email Generation Log - {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, solution in enumerate(email_solutions, 1):
                    f.write(f"EMAIL {i}:\n")
                    f.write(f"Confidence: {solution.confidence_score:.2f}\n")
                    f.write(f"Repos analyzed: {len(solution.github_actions)}\n")
                    f.write(f"Original query preview: {solution.original_user_query[:100]}...\n")
                    f.write("\n")
        except Exception as e:
            self.logger.error(f"Failed to save generation log: {e}")
