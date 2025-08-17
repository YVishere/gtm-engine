"""Enhanced Purpose Detection Engine with deep analysis and transparency."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from models import ProcessedContent
from llm_integration import RAGLLMIntegration


@dataclass
class OpportunityAnalysis:
    """Deep analysis of opportunity context"""
    extracted_technologies: List[str]
    problem_type: str
    urgency_analysis: str
    business_context: str
    technical_complexity: int  # 1-10 scale
    solution_requirements: List[str]
    confidence: float


@dataclass
class EnhancedRAGPurpose:
    """Enhanced RAG purpose with reasoning and predictions"""
    primary_purpose: str
    technologies: List[str]
    search_strategy: str
    urgency_context: str
    reasoning: OpportunityAnalysis
    expected_repositories: List[str]
    expected_file_patterns: List[str]
    success_criteria: Dict[str, Any]
    confidence_score: float


@dataclass
class PredictionSet:
    """Predictions about what the search should find"""
    expected_repo_types: List[str]
    expected_technologies: List[str]
    minimum_relevant_repos: int
    ideal_file_patterns: List[str]
    success_indicators: List[str]


class TransparentRAGPurposeEngine:
    """Enhanced purpose detection with full transparency and reasoning"""
    
    def __init__(self, llm_integration: RAGLLMIntegration):
        self.llm_integration = llm_integration
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_opportunity_context(self, opportunity: ProcessedContent) -> OpportunityAnalysis:
        """Deep analysis of the opportunity context"""
        
        question_text = f"{opportunity.original.title} {opportunity.original.content}"
        
        # Extract technologies with confidence scoring
        technologies = self.extract_technologies_with_confidence(question_text, opportunity.key_topics)
        
        # Classify problem type
        problem_type = self.classify_problem_type(question_text, opportunity.key_topics)
        
        # Analyze urgency context
        urgency_analysis = self.analyze_urgency_context(opportunity, question_text)
        
        # Extract business context
        business_context = self.extract_business_context(question_text, opportunity.original.source.value)
        
        # Assess technical complexity
        technical_complexity = self.assess_technical_complexity(technologies, problem_type, question_text)
        
        # Identify solution requirements
        solution_requirements = self.identify_solution_requirements(question_text, problem_type)
        
        return OpportunityAnalysis(
            extracted_technologies=technologies,
            problem_type=problem_type,
            urgency_analysis=urgency_analysis,
            business_context=business_context,
            technical_complexity=technical_complexity,
            solution_requirements=solution_requirements,
            confidence=self.calculate_analysis_confidence(technologies, problem_type, question_text)
        )
    
    def extract_technologies_with_confidence(self, question_text: str, topics: List[str]) -> List[str]:
        """Extract technologies with confidence scoring - FIXED: No authentication bias"""
        
        # Comprehensive technology patterns without bias
        tech_patterns = {
            'webassembly': {
                'keywords': ['webassembly', 'wasm', 'emscripten', 'webasm'],
                'weight': 1.0
            },
            'react': {
                'keywords': ['react', 'reactjs', 'jsx', 'react native', 'hooks', 'useState', 'useEffect'],
                'weight': 1.0
            },
            'angular': {
                'keywords': ['angular', '@angular', 'ng-', 'angular.js', 'angularjs'],
                'weight': 1.0
            },
            'vue': {
                'keywords': ['vue', 'vuejs', 'vue.js', 'nuxt', 'composition api'],
                'weight': 1.0
            },
            'astro': {
                'keywords': ['astro', 'astro.js', 'astro build', 'astro components'],
                'weight': 1.0
            },
            'node': {
                'keywords': ['node.js', 'nodejs', 'express', 'npm', 'node server'],
                'weight': 1.0
            },
            'python': {
                'keywords': ['python', 'django', 'flask', 'fastapi', 'pip', 'python3'],
                'weight': 1.0
            },
            'javascript': {
                'keywords': ['javascript', 'js', 'es6', 'es2015', 'babel', 'webpack'],
                'weight': 0.8  # Lower weight as it's common
            },
            'typescript': {
                'keywords': ['typescript', 'ts', '.ts', 'tsc'],
                'weight': 1.0
            },
            'cloudflare': {
                'keywords': ['cloudflare', 'cloudflare pages', 'cloudflare workers', 'cf pages'],
                'weight': 1.0
            },
            'd3': {
                'keywords': ['d3.js', 'd3', 'data visualization', 'd3 charts'],
                'weight': 1.0
            },
            'nextjs': {
                'keywords': ['nextjs', 'next.js', 'vercel', 'ssr'],
                'weight': 1.0
            },
            'docker': {
                'keywords': ['docker', 'container', 'kubernetes', 'k8s'],
                'weight': 1.0
            },
            # Auth technologies (only detected when actually mentioned)
            'jwt': {
                'keywords': ['jwt', 'json web token', 'jsonwebtoken', 'bearer token'],
                'weight': 1.0
            },
            'oauth': {
                'keywords': ['oauth', 'oauth2', 'openid', 'oidc', 'authorization code'],
                'weight': 1.0
            }
        }
        
        # Auth indicators - only add auth-general if these are actually present
        auth_indicators = [
            'authentication', 'authorization', 'login', 'signin', 'signup', 
            'auth', 'security', 'password', 'session', 'cookie'
        ]
        
        if not question_text:
            return ['general']
            
        text_lower = question_text.lower()
        detected_techs = {}
        
        # Pattern-based detection with confidence scoring
        for tech_name, config in tech_patterns.items():
            score = 0.0
            
            # Keyword matching
            for keyword in config['keywords']:
                if keyword.lower() in text_lower:
                    score += config['weight']
            
            if score > 0:
                detected_techs[tech_name] = score
        
        # Sort by confidence score
        sorted_techs = sorted(detected_techs.items(), key=lambda x: x[1], reverse=True)
        final_techs = [tech for tech, score in sorted_techs[:5]]  # Top 5
        
        # CRITICAL FIX: Only add auth-general if auth terms are actually present
        has_auth_terms = any(auth_term in text_lower for auth_term in auth_indicators)
        if has_auth_terms and 'auth-general' not in final_techs:
            final_techs.append('auth-general')
        
        # Add topics that look like technologies (but filter out common words)
        common_words = {'help', 'question', 'problem', 'issue', 'error', 'general', 'work', 'use'}
        for topic in topics:
            topic_clean = topic.lower().strip()
            if (topic_clean not in final_techs and 
                topic_clean not in common_words and 
                len(topic_clean) > 2 and
                topic_clean not in [t.lower() for t in final_techs]):
                final_techs.append(topic)
        
        result = final_techs if final_techs else ['general']
        
        # Log for validation
        print(f"🔧 FIXED TECH EXTRACTION: '{question_text[:50]}...' -> {result}")
        
        return result
    
    def classify_problem_type(self, question_text: str, topics: List[str]) -> str:
        """FIXED: Classify problem type without authentication bias"""
        
        text_lower = question_text.lower()
        
        # Problem type classifiers with weights
        problem_classifiers = {
            'authentication_help': {
                'keywords': ['authentication', 'login', 'signin', 'jwt', 'oauth', 'auth', 'session', 'security'],
                'weight': 1.0
            },
            'implementation_showcase': {
                'keywords': ['sharing', 'built', 'created', 'made', 'introducing', 'show', 'demo', 'embedding'],
                'weight': 1.0
            },
            'project_feedback': {
                'keywords': ['feedback', 'thoughts', 'opinions', 'review', 'critique', 'suggestions'],
                'weight': 1.0
            },
            'debugging_issue': {
                'keywords': ['error', 'broken', 'not working', 'issue', 'problem', 'bug', 'failed'],
                'weight': 1.0
            },
            'implementation_guidance': {
                'keywords': ['how to', 'implement', 'setup', 'configure', 'build', 'create'],
                'weight': 0.8
            },
            'performance_optimization': {
                'keywords': ['performance', 'optimization', 'speed', 'slow', 'optimize', 'faster'],
                'weight': 1.0
            },
            'design_help': {
                'keywords': ['design', 'ui', 'ux', 'layout', 'styling', 'css', 'visual'],
                'weight': 1.0
            },
            'architectural_decision': {
                'keywords': ['best practice', 'recommend', 'should i', 'which', 'advice', 'guidance'],
                'weight': 0.8
            }
        }
        
        scores = {}
        for problem_type, config in problem_classifiers.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in text_lower:
                    score += config['weight']
            scores[problem_type] = score
        
        # Return highest scoring type, or general if no clear match
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            result = best_type[0] if best_type[1] > 0 else 'general_consultation'
        else:
            result = 'general_consultation'
        
        # Log for validation
        print(f"🎯 FIXED PROBLEM TYPE: '{question_text[:50]}...' -> {result}")
        
        return result
    
    def analyze_urgency_context(self, opportunity: ProcessedContent, question_text: str) -> str:
        """Analyze the urgency context with reasoning"""
        
        urgency = opportunity.urgency_level
        text_lower = question_text.lower()
        
        urgency_indicators = {
            'high': ['urgent', 'asap', 'production', 'broken', 'critical', 'emergency'],
            'medium': ['soon', 'planning', 'project', 'deadline', 'implementing'],
            'low': ['curious', 'learning', 'exploring', 'considering', 'future']
        }
        
        detected_indicators = []
        for level, indicators in urgency_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    detected_indicators.append(f"{level}: {indicator}")
        
        context = f"Urgency Level: {urgency.upper()}"
        if detected_indicators:
            context += f" (Indicators: {', '.join(detected_indicators)})"
        
        return context
    
    def extract_business_context(self, question_text: str, source: str) -> str:
        """Extract business context from the question"""
        
        text_lower = question_text.lower()
        
        if any(word in text_lower for word in ['startup', 'small company', 'indie']):
            business_size = 'startup'
        elif any(word in text_lower for word in ['enterprise', 'corporate', 'large company']):
            business_size = 'enterprise'
        else:
            business_size = 'unknown'
        
        if any(word in text_lower for word in ['team', 'developers', 'colleagues']):
            team_context = 'team_development'
        else:
            team_context = 'individual_development'
        
        return f"Business Context: {business_size} {team_context} via {source}"
    
    def assess_technical_complexity(self, technologies: List[str], problem_type: str, question_text: str) -> int:
        """Assess technical complexity on 1-10 scale"""
        
        complexity = 3  # Base complexity
        
        # Technology complexity
        complex_techs = ['oauth', 'saml', 'openid', 'kubernetes', 'microservices']
        complexity += sum(1 for tech in technologies if tech in complex_techs)
        
        # Problem type complexity
        complex_problems = ['scaling_challenge', 'security_concern', 'migration_evaluation']
        if problem_type in complex_problems:
            complexity += 2
        
        # Integration complexity
        if len(technologies) > 3:
            complexity += 1
        
        # Text indicators
        text_lower = question_text.lower()
        if any(word in text_lower for word in ['distributed', 'microservices', 'multi-tenant']):
            complexity += 2
        
        return min(complexity, 10)
    
    def identify_solution_requirements(self, question_text: str, problem_type: str) -> List[str]:
        """Identify what the solution needs to provide"""
        
        requirements = []
        text_lower = question_text.lower()
        
        # Based on problem type
        if problem_type == 'debugging_issue':
            requirements.extend(['working code examples', 'error handling patterns', 'troubleshooting guides'])
        elif problem_type == 'implementation_guidance':
            requirements.extend(['step-by-step tutorials', 'best practices', 'starter templates'])
        elif problem_type == 'scaling_challenge':
            requirements.extend(['production examples', 'performance optimization', 'architecture patterns'])
        
        # Based on text content
        if 'refresh token' in text_lower:
            requirements.append('token refresh mechanisms')
        if 'logout' in text_lower:
            requirements.append('secure logout patterns')
        if 'middleware' in text_lower:
            requirements.append('middleware implementations')
        
        return requirements[:5]
    
    def calculate_analysis_confidence(self, technologies: List[str], problem_type: str, question_text: str) -> float:
        """Calculate confidence in the analysis"""
        
        confidence = 0.5  # Base confidence
        
        # Technology detection confidence
        if len(technologies) > 0:
            confidence += 0.2
        if len(technologies) > 2:
            confidence += 0.1
        
        # Problem clarity
        if problem_type != 'general_consultation':
            confidence += 0.2
        
        # Question length and detail
        if len(question_text) > 100:
            confidence += 0.1
        if len(question_text) > 300:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def generate_purpose_with_reasoning(self, opportunity: ProcessedContent) -> EnhancedRAGPurpose:
        """Generate enhanced purpose with full reasoning"""
        
        # Step 1: Deep opportunity analysis
        analysis = self.analyze_opportunity_context(opportunity)
        
        print(f"   🧠 Deep Analysis Complete:")
        print(f"      🔧 Technologies: {', '.join(analysis.extracted_technologies)}")
        print(f"      📋 Problem Type: {analysis.problem_type}")
        print(f"      ⚡ Urgency: {analysis.urgency_analysis}")
        print(f"      🏢 Business: {analysis.business_context}")
        print(f"      🎯 Complexity: {analysis.technical_complexity}/10")
        print(f"      📊 Confidence: {analysis.confidence:.2f}")
        
        # Step 2: Generate LLM-enhanced purpose
        llm_purpose = self.generate_llm_purpose(opportunity, analysis)
        
        # Step 3: Create predictions
        predictions = self.generate_predictions(analysis, llm_purpose)
        
        # Step 4: Define success criteria
        success_criteria = self.define_success_criteria(analysis, predictions)
        
        enhanced_purpose = EnhancedRAGPurpose(
            primary_purpose=llm_purpose.get('primary_purpose', f'{analysis.extracted_technologies[0] if analysis.extracted_technologies else "authentication"} {analysis.problem_type}'),
            technologies=llm_purpose.get('technologies', analysis.extracted_technologies),
            search_strategy=llm_purpose.get('search_strategy', f'Find {analysis.problem_type} solutions'),
            urgency_context=llm_purpose.get('urgency_context', analysis.urgency_analysis),
            reasoning=analysis,
            expected_repositories=predictions.expected_repo_types,
            expected_file_patterns=predictions.ideal_file_patterns,
            success_criteria=success_criteria,
            confidence_score=(analysis.confidence + 0.3) / 2  # Blend analysis and LLM confidence
        )
        
        return enhanced_purpose
    
    def generate_llm_purpose(self, opportunity: ProcessedContent, analysis: OpportunityAnalysis) -> Dict[str, Any]:
        """FIXED: Generate purpose using LLM without authentication bias"""
        
        # Determine the actual topic without bias
        content_text = f"{opportunity.original.title} {opportunity.original.content}"
        is_auth_related = any(auth_term in content_text.lower() for auth_term in 
                             ['authentication', 'login', 'signin', 'jwt', 'oauth', 'auth', 'session'])
        
        enhanced_prompt = f"""You are a technical research assistant specializing in developer problems.

OPPORTUNITY ANALYSIS:
- Technologies: {', '.join(analysis.extracted_technologies)}
- Problem Type: {analysis.problem_type}
- Technical Complexity: {analysis.technical_complexity}/10
- Solution Requirements: {', '.join(analysis.solution_requirements)}
- Business Context: {analysis.business_context}
- Is Authentication Related: {is_auth_related}

ORIGINAL QUESTION:
{opportunity.original.title}
{opportunity.original.content[:500]}

Based on this analysis, determine the optimal GitHub search strategy. 

IMPORTANT: Focus on what the user actually needs help with, not authentication unless explicitly mentioned.

1. PRIMARY_PURPOSE: What the user actually wants to accomplish (not authentication unless explicitly asked)
2. TECHNOLOGIES: Key tech stack from analysis
3. SEARCH_STRATEGY: How to find relevant repositories for their actual problem
4. URGENCY_CONTEXT: Priority context

Return ONLY valid JSON:
{{
    "primary_purpose": "Specific goal based on actual problem: {analysis.problem_type}",
    "technologies": {analysis.extracted_technologies},
    "search_strategy": "Find {analysis.problem_type} solutions for {analysis.extracted_technologies[0] if analysis.extracted_technologies else 'general'}",
    "urgency_context": "{analysis.urgency_analysis}"
}}"""
        
        try:
            return self.llm_integration.generate_purpose_with_llm(enhanced_prompt)
        except Exception as e:
            self.logger.warning(f"LLM purpose generation failed: {e}")
            return {}
    
    def generate_predictions(self, analysis: OpportunityAnalysis, llm_purpose: Dict[str, Any]) -> PredictionSet:
        """FIXED: Generate predictions based on actual problem type, not always auth"""
        
        # Predict repository types based on actual problem type
        repo_types = []
        
        # Check if authentication is actually relevant
        is_auth_problem = analysis.problem_type in ['authentication_help', 'security_concern'] or \
                         any(auth_term in analysis.extracted_technologies for auth_term in ['jwt', 'oauth', 'auth-general'])
        
        if is_auth_problem:
            # Only add auth-specific repos if actually needed
            for tech in analysis.extracted_technologies:
                if tech in ['react', 'vue', 'angular']:
                    repo_types.extend([f'{tech}-auth-examples', f'{tech}-jwt-implementation'])
                elif tech in ['node', 'express']:
                    repo_types.extend(['nodejs-auth-middleware', 'express-authentication'])
                elif tech in ['python', 'django', 'flask']:
                    repo_types.extend(['python-auth-libraries', f'{tech}-authentication'])
        else:
            # Generate context-appropriate repository types
            for tech in analysis.extracted_technologies:
                if tech == 'webassembly':
                    repo_types.extend(['webassembly-examples', 'emscripten-tutorials', 'wasm-integration'])
                elif tech == 'astro':
                    repo_types.extend(['astro-examples', 'astro-build-examples', 'astro-components'])
                elif tech in ['react', 'vue', 'angular']:
                    repo_types.extend([f'{tech}-examples', f'{tech}-tutorials', f'{tech}-best-practices'])
                elif tech in ['node', 'express']:
                    repo_types.extend(['nodejs-examples', 'express-tutorials', 'backend-examples'])
                elif tech == 'python':
                    repo_types.extend(['python-examples', 'python-tutorials', 'python-projects'])
                elif tech == 'd3':
                    repo_types.extend(['d3-examples', 'data-visualization', 'd3-tutorials'])
                elif tech == 'cloudflare':
                    repo_types.extend(['cloudflare-examples', 'cloudflare-pages', 'static-site-deployment'])
        
        # Generate context-appropriate file patterns
        file_patterns = ['README.md', 'package.json', 'docs/']
        
        if is_auth_problem:
            file_patterns.extend(['auth.js', 'authentication.py', 'login.jsx'])
        else:
            # Add patterns based on technologies and problem type
            for tech in analysis.extracted_technologies:
                if tech == 'react':
                    file_patterns.extend(['App.jsx', 'components/', 'src/'])
                elif tech == 'astro':
                    file_patterns.extend(['astro.config.js', 'src/pages/', 'src/components/'])
                elif tech == 'webassembly':
                    file_patterns.extend(['*.wasm', 'build.js', 'emscripten/'])
                elif tech == 'd3':
                    file_patterns.extend(['chart.js', 'visualization.js', 'data/'])
                elif tech == 'python':
                    file_patterns.extend(['main.py', 'requirements.txt', 'setup.py'])
        
        # Success indicators based on actual problem
        success_indicators = [
            f'Find examples for {analysis.problem_type}',
            f'Locate {len(analysis.extracted_technologies)} technology-specific solutions',
            f'Discover implementations matching requirements'
        ]
        
        return PredictionSet(
            expected_repo_types=repo_types[:6],
            expected_technologies=analysis.extracted_technologies,
            minimum_relevant_repos=max(2, len(analysis.extracted_technologies)),
            ideal_file_patterns=file_patterns[:8],
            success_indicators=success_indicators
        )
    
    def define_success_criteria(self, analysis: OpportunityAnalysis, predictions: PredictionSet) -> Dict[str, Any]:
        """Define clear success criteria for the search"""
        
        return {
            'minimum_repositories': predictions.minimum_relevant_repos,
            'required_technologies': predictions.expected_technologies,
            'target_file_types': ['code_examples', 'documentation', 'tutorials'],
            'quality_thresholds': {
                'min_stars': 10 if analysis.technical_complexity > 5 else 5,
                'min_relevance_score': 0.6,
                'max_age_days': 365 * 2  # 2 years
            },
            'solution_coverage': {
                'must_address_problem_type': analysis.problem_type,
                'should_include_requirements': analysis.solution_requirements[:3],
                'complexity_appropriate': analysis.technical_complexity
            }
        }
