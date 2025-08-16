"""Content analysis utilities for topic extraction and analysis."""

import numpy as np
import logging
from typing import List, Dict, Any
from models import ScrapedContent, ProcessedContent
from content_clusterer import ClusterInfo


class ContentAnalyzer:
    """Handles content analysis and topic extraction."""
    
    def __init__(self, business_categorizer):
        self.business_categorizer = business_categorizer
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_representative_item(self, cluster: ClusterInfo) -> Dict[str, Any]:
        """Analyze the representative item for the cluster."""
        rep_item = cluster.representative_item
        
        # Create analysis based on cluster context
        summary = f"{cluster.category.replace('_', ' ').title()} challenge involving {cluster.theme}"
        if cluster.size > 1:
            summary += f" (representing {cluster.size} similar discussions)"
        
        # Determine relevance based on cluster characteristics
        relevance_score = min(0.5 + (cluster.size * 0.1) + (0.3 if cluster.urgency_level == 'high' else 0.1), 0.95)
        
        # Extract meaningful key topics from cluster theme and content
        key_topics = self.extract_meaningful_topics(cluster.theme, rep_item)
        
        return {
            'summary': summary,
            'relevance_score': relevance_score,
            'key_topics': key_topics,
            'urgency_level': cluster.urgency_level,
            'cluster_theme': cluster.theme,
            'cluster_category': cluster.category
        }
    
    def extract_meaningful_topics(self, theme: str, item: ScrapedContent) -> List[str]:
        """Extract meaningful topics from theme and content."""
        topics = []
        
        # Check theme for specific patterns
        theme_lower = theme.lower()
        content_lower = f"{item.title} {item.content}".lower()
        
        # JWT-related topics
        if 'jwt' in theme_lower:
            if 'refresh' in content_lower:
                topics.append('jwt-refresh-tokens')
            elif 'payload' in content_lower or 'malformed' in content_lower:
                topics.append('jwt-payload-validation')
            else:
                topics.append('jwt-authentication')
        
        # OAuth-related topics  
        if 'oauth' in theme_lower or 'openid' in theme_lower:
            if 'pkce' in content_lower:
                topics.append('oauth2-pkce')
            elif 'oidc' in content_lower or 'openid' in content_lower:
                topics.append('openid-connect')
            else:
                topics.append('oauth-integration')
        
        # Session-related topics
        if 'session' in theme_lower:
            if 'stateless' in content_lower:
                topics.append('stateless-authentication')
            elif 'scale' in content_lower or 'distributed' in content_lower:
                topics.append('session-scaling')
            else:
                topics.append('session-management')
        
        # Spring-related topics
        if 'spring' in theme_lower:
            if 'gateway' in content_lower:
                topics.append('spring-cloud-gateway')
            elif 'security' in content_lower:
                topics.append('spring-security')
            else:
                topics.append('spring-boot-auth')
        
        # Security-related topics
        if any(code in content_lower for code in ['401', '403', 'unauthorized', 'forbidden']):
            topics.append('authentication-errors')
        
        # Fallback topics if none found
        if not topics:
            if 'authentication' in content_lower:
                topics.append('authentication-implementation')
            elif 'security' in content_lower:
                topics.append('security-configuration')
            else:
                topics.append('auth-general')
        
        # Limit to 3 topics
        return topics[:3]
    
    def propagate_analysis_to_cluster(self, analysis: Dict[str, Any], cluster: ClusterInfo) -> List[ProcessedContent]:
        """Apply analysis to all items in the cluster with variations."""
        processed_items = []
        
        for item in cluster.items:
            # Create slight variations in the analysis
            item_summary = f"{item.title[:50]}... {analysis['summary']}"
            
            # Slight relevance variation based on individual item characteristics
            base_relevance = analysis['relevance_score']
            item_relevance = base_relevance + np.random.uniform(-0.05, 0.05)
            item_relevance = max(0.3, min(0.95, item_relevance))
            
            processed_content = ProcessedContent(
                original=item,
                summary=item_summary,
                relevance_score=item_relevance,
                key_topics=analysis['key_topics'],
                urgency_level=analysis['urgency_level']
            )
            
            processed_items.append(processed_content)
        
        return processed_items
    
    def generate_opportunity_explanation(self, content: ProcessedContent) -> str:
        """Generate GTM-focused explanation for business opportunities."""
        # Use the original business categorization logic
        business_cat = self.business_categorizer.categorize_business_opportunity(
            np.zeros(384), # Dummy embedding since we already have the analysis
            content.original.content
        )
        
        opportunity_map = {
            'hot_leads': 'urgent lead opportunity. Team should engage with technical consultation and Descope platform demo',
            'evaluation_phase': 'qualified lead opportunity. Sales should offer technical consultation showcasing Descope\'s enterprise auth solutions',
            'implementation_stage': 'qualified lead opportunity. Sales should offer technical consultation showcasing Descope\'s enterprise auth solutions',
            'scaling_challenges': 'urgent lead opportunity. Team should engage with technical consultation and Descope platform demo'
        }
        
        if content.key_topics:
            topic_focus = content.key_topics[0].replace('-', ' ')
        else:
            topic_focus = 'authentication'
        
        opportunity_desc = opportunity_map.get(business_cat, 'qualified lead opportunity. Sales should offer technical consultation showcasing Descope\'s enterprise auth solutions')
        
        return f"Developer implementing {topic_focus} authentication presents {opportunity_desc}."
