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
        """Extract technologies with confidence scoring"""
        
        tech_indicators = {
            'react': ['react', 'reactjs', 'jsx', 'hooks', 'components'],
            'vue': ['vue', 'vuejs', 'nuxt', 'composition api'],
            'angular': ['angular', 'ng', 'typescript', 'rxjs'],
            'node': ['node', 'nodejs', 'express', 'npm', 'server'],
            'python': ['python', 'django', 'flask', 'fastapi', 'pip'],
            'java': ['java', 'spring', 'springboot', 'maven', 'gradle'],
            'javascript': ['javascript', 'js', 'es6', 'babel', 'webpack'],
            'jwt': ['jwt', 'jsonwebtoken', 'token', 'bearer'],
            'oauth': ['oauth', 'oauth2', 'openid', 'oidc', 'authorization'],
            'nextjs': ['nextjs', 'next.js', 'vercel', 'ssr'],
            'docker': ['docker', 'container', 'kubernetes', 'k8s']
        }
        
        detected_techs = []
        text_lower = question_text.lower()
        
        for tech, indicators in tech_indicators.items():
            confidence = sum(1 for indicator in indicators if indicator in text_lower)
            if confidence > 0:
                detected_techs.append(tech)
        
        # Add topics that look like technologies
        for topic in topics:
            if topic.lower() not in [t.lower() for t in detected_techs] and len(topic) > 2:
                detected_techs.append(topic)
        
        return detected_techs[:5]  # Top 5 most relevant
    
    def classify_problem_type(self, question_text: str, topics: List[str]) -> str:
        """Classify the type of authentication problem"""
        
        text_lower = question_text.lower()
        
        if any(word in text_lower for word in ['not working', 'broken', 'error', 'failed']):
            return 'debugging_issue'
        elif any(word in text_lower for word in ['how to', 'implement', 'setup', 'configure']):
            return 'implementation_guidance'
        elif any(word in text_lower for word in ['best practice', 'recommend', 'should i', 'which']):
            return 'architectural_decision'
        elif any(word in text_lower for word in ['scale', 'performance', 'optimize', 'production']):
            return 'scaling_challenge'
        elif any(word in text_lower for word in ['secure', 'security', 'vulnerability', 'attack']):
            return 'security_concern'
        elif any(word in text_lower for word in ['migrate', 'switch', 'replace', 'alternative']):
            return 'migration_evaluation'
        else:
            return 'general_consultation'
    
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
        """Generate purpose using LLM with analysis context"""
        
        enhanced_prompt = f"""You are a technical research assistant with deep authentication expertise.

OPPORTUNITY ANALYSIS:
- Technologies: {', '.join(analysis.extracted_technologies)}
- Problem Type: {analysis.problem_type}
- Technical Complexity: {analysis.technical_complexity}/10
- Solution Requirements: {', '.join(analysis.solution_requirements)}
- Business Context: {analysis.business_context}

ORIGINAL QUESTION:
{opportunity.original.title}
{opportunity.original.content[:500]}

Based on this analysis, determine the optimal GitHub search strategy:

1. PRIMARY_PURPOSE: Specific technical goal (be very specific based on problem type)
2. TECHNOLOGIES: Key tech stack (from analysis)
3. SEARCH_STRATEGY: How to search (focus on solution requirements)
4. URGENCY_CONTEXT: Priority context

Return ONLY valid JSON:
{{
    "primary_purpose": "Specific goal based on {analysis.problem_type}",
    "technologies": {analysis.extracted_technologies},
    "search_strategy": "Strategy targeting {', '.join(analysis.solution_requirements[:2])}",
    "urgency_context": "{analysis.urgency_analysis}"
}}"""
        
        try:
            return self.llm_integration.generate_purpose_with_llm(enhanced_prompt)
        except Exception as e:
            self.logger.warning(f"LLM purpose generation failed: {e}")
            return {}
    
    def generate_predictions(self, analysis: OpportunityAnalysis, llm_purpose: Dict[str, Any]) -> PredictionSet:
        """Generate predictions about expected search results"""
        
        # Predict repository types based on technologies and problem type
        repo_types = []
        for tech in analysis.extracted_technologies:
            if tech in ['react', 'vue', 'angular']:
                repo_types.extend([f'{tech}-auth-examples', f'{tech}-jwt-implementation'])
            elif tech in ['node', 'express']:
                repo_types.extend(['nodejs-auth-middleware', 'express-authentication'])
            elif tech in ['python', 'django', 'flask']:
                repo_types.extend(['python-auth-libraries', f'{tech}-authentication'])
        
        # File patterns based on technologies
        file_patterns = ['README.md', 'auth.js', 'authentication.py', 'login.jsx']
        for tech in analysis.extracted_technologies:
            if tech == 'react':
                file_patterns.extend(['AuthContext.js', 'useAuth.js', 'ProtectedRoute.jsx'])
            elif tech == 'jwt':
                file_patterns.extend(['jwt.js', 'token.js', 'refresh.js'])
        
        # Success indicators
        success_indicators = [
            f'Find {len(analysis.extracted_technologies)} technology-specific examples',
            f'Locate solutions for {analysis.problem_type}',
            f'Discover patterns matching {len(analysis.solution_requirements)} requirements'
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
