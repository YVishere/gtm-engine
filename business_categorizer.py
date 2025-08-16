"""Business categorizer for content classification and opportunity analysis."""

import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity


class BusinessCategorizer:
    """Categorizes content for business intelligence."""
    
    BUSINESS_CATEGORIES = {
        'hot_leads': {
            'indicators': ['urgent', 'help needed', 'not working', 'broken', 'error', 'issue', 'problem'],
            'urgency': 'high',
            'opportunity': 'immediate_consultation'
        },
        'evaluation_phase': {
            'indicators': ['comparing', 'which is better', 'recommendations', 'vs', 'choose', 'decide'],
            'urgency': 'medium', 
            'opportunity': 'competitive_positioning'
        },
        'implementation_stage': {
            'indicators': ['how to implement', 'best practices', 'tutorial', 'guide', 'setup', 'configure'],
            'urgency': 'medium',
            'opportunity': 'technical_support'
        },
        'scaling_challenges': {
            'indicators': ['performance', 'scale', 'enterprise', 'production', 'microservices', 'distributed'],
            'urgency': 'high',
            'opportunity': 'enterprise_solution'
        }
    }
    
    CLUSTER_CATEGORIES = {
        'jwt_issues': ['jwt', 'token', 'payload', 'malformed', 'validation', 'refresh'],
        'oauth_integration': ['oauth', 'openid', 'pkce', 'authorization', 'oidc'],
        'session_management': ['session', 'cookie', 'stateless', 'persistence', 'jsessionid'],
        'framework_specific': ['asp.net', 'spring', 'fastapi', 'react', 'node', 'blazor'],
        'enterprise_auth': ['saml', 'sso', 'enterprise', 'ldap', 'active directory'],
        'security_concerns': ['401', '403', 'unauthorized', 'vulnerability', 'security'],
        'implementation_help': ['how to', 'implement', 'setup', 'configure', 'tutorial']
    }
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self._category_embeddings = self._precompute_category_embeddings()
    
    def _precompute_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Precompute embeddings for all categories."""
        embeddings = {}
        
        # Business categories
        for category, config in self.BUSINESS_CATEGORIES.items():
            text = ' '.join(config['indicators'])
            embeddings[f"business_{category}"] = self.embedding_model.encode([text])[0]
        
        # Technical categories
        for category, keywords in self.CLUSTER_CATEGORIES.items():
            text = ' '.join(keywords)
            embeddings[f"technical_{category}"] = self.embedding_model.encode([text])[0]
        
        return embeddings
    
    def categorize_business_opportunity(self, content_embedding: np.ndarray, content_text: str) -> str:
        """Categorize content for business opportunity."""
        category_scores = {}
        
        for category, config in self.BUSINESS_CATEGORIES.items():
            # Semantic similarity score
            cat_embedding = self._category_embeddings[f"business_{category}"]
            semantic_score = cosine_similarity([content_embedding], [cat_embedding])[0][0]
            
            # Keyword presence score
            keyword_score = sum(1 for kw in config['indicators'] if kw.lower() in content_text.lower())
            keyword_score = min(keyword_score / len(config['indicators']), 1.0)  # Normalize
            
            category_scores[category] = semantic_score * 0.7 + keyword_score * 0.3
        
        return max(category_scores, key=category_scores.get)
    
    def categorize_technical_cluster(self, cluster_topics: List[str], cluster_texts: List[str]) -> str:
        """Categorize a cluster based on technical content."""
        # Combine all cluster text for analysis
        combined_text = ' '.join(cluster_texts).lower()
        cluster_embedding = self.embedding_model.encode([combined_text])[0]
        
        category_scores = {}
        for category, keywords in self.CLUSTER_CATEGORIES.items():
            # Semantic similarity
            cat_embedding = self._category_embeddings[f"technical_{category}"]
            semantic_score = cosine_similarity([cluster_embedding], [cat_embedding])[0][0]
            
            # Topic overlap score
            topic_overlap = len(set(cluster_topics) & set(keywords)) / len(keywords)
            
            category_scores[category] = semantic_score * 0.6 + topic_overlap * 0.4
        
        return max(category_scores, key=category_scores.get)
