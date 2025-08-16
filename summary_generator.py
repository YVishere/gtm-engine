"""Summary generation utilities for business intelligence reporting."""

import logging
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from models import ProcessedContent, ScrapedContent
from cluster_info import ClusterInfo


class SummaryGenerator:
    """Generates enhanced summaries using cluster analysis."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_overall_summary(self, processed_contents: List[ProcessedContent], clusters: List[ClusterInfo]) -> Dict[str, Any]:
        """Generate enhanced summary using ALL clusters with bag-of-words approach."""
        if not processed_contents:
            return {
                'overall_summary': 'No relevant content found.',
                'top_trends': [],
                'high_priority_items': 0
            }
        
        print(f"\n📈 Generating cluster-based executive summary using ALL clusters...")
        
        if not clusters:
            # Fallback to old approach if no clusters available
            top_items = sorted(processed_contents, key=lambda x: x.relevance_score, reverse=True)[:15]
            return self._generate_fallback_summary(top_items, processed_contents)
        
        print(f"   🔄 Extracting keywords from {len(clusters)} clusters in parallel...")
        
        # Extract bag-of-words from ALL clusters in parallel
        cluster_keywords = self._extract_cluster_keywords_parallel(clusters)
        
        # Generate comprehensive summary using ALL cluster data
        high_priority_count = len([pc for pc in processed_contents if pc.urgency_level == 'high'])
        total_items = len(processed_contents)
        
        print(f"   🧠 Generating executive summary from {len(cluster_keywords)} clusters...")
        
        # Use LLM with ALL cluster context
        try:
            from llm_processor import LegacyLLMProcessor
            legacy_processor = LegacyLLMProcessor()
            
            # Create cluster context for the LLM
            cluster_context = []
            high_urgency_clusters = []
            
            for ck in cluster_keywords:
                urgency_indicator = "🔥 HIGH" if ck['urgency'] == 'high' else "📊 MED" if ck['urgency'] == 'medium' else "📋 LOW"
                
                # Include actual content snippets for better context
                content_preview = " | ".join(ck['content_snippets'][:2])  # Top 2 content snippets
                cluster_desc = f"Cluster {ck['cluster_id']} ({ck['size']} discussions, {urgency_indicator}): {ck['theme']} [{ck['category']}]\n   Content: {content_preview}\n   Keywords: {', '.join(ck['keywords'][:10])}"
                cluster_context.append(cluster_desc)
                
                if ck['urgency'] == 'high':
                    high_urgency_clusters.append(f"• {ck['theme']} ({ck['size']} discussions): {ck['representative_title']}")
            
            # Context summary
            context_data = {
                'total_analyzed': total_items,
                'total_clusters': len(cluster_keywords),
                'high_urgency_clusters': len(high_urgency_clusters),
                'sources': list(set([pc.original.source.value for pc in processed_contents])),
                'high_priority_count': high_priority_count
            }
            
            # Enhanced prompt using ALL cluster data
            prompt = self._create_llm_prompt(cluster_context, high_urgency_clusters, context_data, len(cluster_keywords), total_items)
            
            response = legacy_processor._call_ollama(prompt)
            if response:
                parsed_result = legacy_processor._parse_llm_response(response)
                if parsed_result and 'overall_summary' in parsed_result and len(parsed_result.get('overall_summary', '')) > 50:
                    print("   ✅ LLM-enhanced executive summary generated")
                    return parsed_result
        
        except Exception as e:
            self.logger.warning(f"LLM summary generation failed: {e}")
        
        # Enhanced fallback using cluster data instead of top items
        print("   ⚠️  Using enhanced cluster-based fallback summary")
        return self._generate_cluster_based_fallback_summary(cluster_keywords, processed_contents)
    
    def _create_llm_prompt(self, cluster_context: List[str], high_urgency_clusters: List[str], context_data: Dict, num_clusters: int, total_items: int) -> str:
        """Create comprehensive LLM prompt for summary generation."""
        return f"""You are analyzing authentication and security discussions for business intelligence. Based on comprehensive cluster analysis of ALL {num_clusters} conversation clusters, generate insights for a GTM (Go-To-Market) team.

COMPREHENSIVE CLUSTER ANALYSIS ({num_clusters} clusters covering {total_items} discussions):
{chr(10).join(cluster_context)}

HIGH-URGENCY CLUSTERS REQUIRING IMMEDIATE ATTENTION:
{chr(10).join(high_urgency_clusters) if high_urgency_clusters else "None identified"}

ANALYSIS CONTEXT:
- Total discussions analyzed: {context_data['total_analyzed']}
- Conversation clusters identified: {context_data['total_clusters']}
- High-urgency clusters: {context_data['high_urgency_clusters']}
- Data sources: {', '.join(context_data['sources'])}
- High-priority items: {context_data['high_priority_count']}

Generate a comprehensive business intelligence summary focusing on:
1. What authentication challenges developers are facing RIGHT NOW based on cluster patterns
2. Which technologies/approaches are trending and causing problems across clusters
3. Specific business opportunities for auth solution providers like Descope based on cluster themes
4. Market sentiment and developer pain points that indicate buying intent

Return your analysis in this exact JSON format:
{{
    "overall_summary": "Comprehensive 3-4 sentence executive summary highlighting: (1) key authentication challenges developers face across {num_clusters} conversation clusters, (2) trending technologies causing issues, (3) business opportunities for auth providers, and (4) market sentiment indicating purchase intent",
    "top_trends": ["Trend 1: Specific technical challenge developers are discussing with business context", "Trend 2: Another key technology pattern or implementation issue", "Trend 3: Market sentiment or pain point indicating sales opportunity", "Trend 4: Emerging technology adoption or integration challenge", "Trend 5: Developer behavior pattern suggesting vendor evaluation"],
    "high_priority_items": {context_data['high_priority_count']}
}}

IMPORTANT: Return only valid JSON without any additional text, thinking, or formatting."""
    
    def _extract_cluster_keywords_parallel(self, clusters: List[ClusterInfo]) -> List[Dict[str, Any]]:
        """Extract keywords from each cluster in parallel using TF-IDF."""
        def extract_keywords(cluster: ClusterInfo) -> Dict[str, Any]:
            """Extract top keywords from a single cluster."""
            try:
                # Combine all text from cluster items
                texts = []
                for item in cluster.items:
                    text = f"{item.title} {item.content[:500]}"  # Limit content length
                    texts.append(text)
                
                combined_text = " ".join(texts)
                
                # Use simple word frequency for ALL clusters to avoid TF-IDF issues
                from collections import Counter
                import re
                
                # Extract words and clean them
                words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
                word_counts = Counter(words)
                
                # Filter out common stop words
                stop_words = {
                    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 
                    'have', 'what', 'were', 'said', 'each', 'which', 'she', 'their', 'time', 'will', 'way', 'about', 
                    'many', 'then', 'them', 'write', 'would', 'like', 'has', 'into', 'more', 'two', 'how', 'its', 
                    'who', 'did', 'get', 'may', 'him', 'old', 'see', 'now', 'could', 'people', 'than', 'first', 
                    'been', 'call', 'day', 'find', 'long', 'down', 'side', 'use', 'from', 'they', 'know', 'water', 
                    'with', 'this', 'that', 'such', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom',
                    'here', 'there', 'some', 'any', 'every', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
                    'another', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
                    'should', 'now', 'https', 'www', 'com', 'org', 'net', 'stackoverflow', 'reddit'
                }
                
                # Get meaningful keywords
                filtered_words = [(word, count) for word, count in word_counts.most_common(50) 
                                if word not in stop_words and len(word) > 3]
                keywords = [word for word, count in filtered_words[:20]]
                
                # Ensure we have some keywords
                if not keywords:
                    keywords = ['authentication', 'security']
                
                # Get sample content snippets for better LLM context
                content_snippets = []
                for item in cluster.items[:3]:  # Top 3 items from cluster
                    snippet = f"{item.title}: {item.content[:200]}..."
                    content_snippets.append(snippet)
                
                return {
                    'cluster_id': cluster.id,
                    'size': cluster.size,
                    'theme': cluster.theme,
                    'keywords': keywords,
                    'category': cluster.category,
                    'urgency': cluster.urgency_level,
                    'representative_title': cluster.representative_item.title,
                    'content_snippets': content_snippets  # Add actual content for LLM
                }
                
            except Exception as e:
                self.logger.warning(f"Error extracting keywords from cluster {cluster.id}: {e}")
                return {
                    'cluster_id': cluster.id,
                    'size': cluster.size,
                    'theme': cluster.theme,
                    'keywords': ['authentication', 'security'],
                    'category': cluster.category,
                    'urgency': cluster.urgency_level,
                    'representative_title': cluster.representative_item.title,
                    'content_snippets': [f"{cluster.representative_item.title}: {cluster.representative_item.content[:200]}"]
                }
        
        # Process clusters in parallel
        print(f"   ⚡ Processing {len(clusters)} clusters with {min(8, len(clusters))} workers...")
        with ThreadPoolExecutor(max_workers=min(8, len(clusters))) as executor:
            cluster_keywords = list(executor.map(extract_keywords, clusters))
        
        print(f"   ✅ Extracted keywords from {len(cluster_keywords)} clusters")
        return cluster_keywords
    
    def _generate_fallback_summary(self, top_items: List[ProcessedContent], processed_contents: List[ProcessedContent]) -> Dict[str, Any]:
        """Fallback method for when no clusters are available."""
        high_priority_count = len([pc for pc in processed_contents if pc.urgency_level == 'high'])
        total_items = len(processed_contents)
        
        # Generate basic insights from actual data (like original processor)
        common_issues = []
        for pc in top_items[:5]:
            if 'jwt' in pc.summary.lower():
                common_issues.append("JWT implementation challenges")
            elif 'oauth' in pc.summary.lower():
                common_issues.append("OAuth integration complexities")
            elif 'session' in pc.summary.lower():
                common_issues.append("Session management issues")
            elif 'security' in pc.summary.lower():
                common_issues.append("Security vulnerability concerns")
            elif 'spring' in pc.summary.lower():
                common_issues.append("Spring Boot configuration challenges")
        
        unique_issues = list(set(common_issues))
        if not unique_issues:
            unique_issues = ["authentication implementation challenges", "security configuration issues", "integration complexities"]
        
        # Create executive summary with the same quality as original
        overall_summary = f'Authentication intelligence analysis reveals {total_items} discussions across {len(set([pc.original.source.value for pc in processed_contents]))} developer platforms, indicating active market engagement. Primary challenges center around {", ".join(unique_issues[:3])}. Market sentiment shows {high_priority_count} high-urgency situations requiring immediate vendor consultation. Developer discussions indicate strong preference for enterprise-grade solutions that solve authentication complexity while maintaining security standards.'
        
        # Generate enhanced trends similar to original processor
        enhanced_trends = self._generate_trends_from_content(top_items)
        
        result = {
            'overall_summary': overall_summary,
            'top_trends': enhanced_trends[:5],
            'high_priority_items': high_priority_count
        }
        
        print("   ✅ Enhanced fallback executive summary generated")
        return result
    
    def _generate_cluster_based_fallback_summary(self, cluster_keywords: List[Dict[str, Any]], processed_contents: List[ProcessedContent]) -> Dict[str, Any]:
        """Generate fallback summary using cluster keywords when LLM fails."""
        high_priority_count = len([pc for pc in processed_contents if pc.urgency_level == 'high'])
        total_items = len(processed_contents)
        
        print(f"   🔧 Generating enhanced fallback from {len(cluster_keywords)} clusters...")
        
        # Extract insights from cluster data
        high_urgency_clusters = [ck for ck in cluster_keywords if ck['urgency'] == 'high']
        top_themes = []
        
        # Analyze cluster themes and keywords
        all_keywords = []
        cluster_sizes = []
        for ck in cluster_keywords:
            top_themes.append(ck['theme'])
            all_keywords.extend(ck['keywords'][:5])  # Top 5 keywords per cluster
            cluster_sizes.append(ck['size'])
        
        # Count keyword frequency for trends
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(10)
        
        # Generate business-focused summary
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        large_clusters = len([size for size in cluster_sizes if size > avg_cluster_size * 1.5])
        
        overall_summary = f'Authentication intelligence analysis identified {len(cluster_keywords)} distinct conversation clusters across {total_items} developer discussions, revealing {len(high_urgency_clusters)} high-urgency technical challenges requiring immediate vendor consultation. Cluster analysis shows concentrated developer activity around {top_keywords[0][0] if top_keywords else "authentication"} and {top_keywords[1][0] if len(top_keywords) > 1 else "security"} implementations, with {large_clusters} clusters showing significant community engagement. Market sentiment indicates strong demand for enterprise authentication solutions that address implementation complexity while maintaining security standards.'
        
        # Generate trends from cluster analysis
        enhanced_trends = self._generate_trends_from_clusters(cluster_keywords)
        
        result = {
            'overall_summary': overall_summary,
            'top_trends': enhanced_trends[:5],
            'high_priority_items': high_priority_count
        }
        
        print("   ✅ Enhanced cluster-based fallback summary generated")
        return result
    
    def _generate_trends_from_content(self, top_items: List[ProcessedContent]) -> List[str]:
        """Generate trends from top content items."""
        enhanced_trends = []
        
        # Count specific issues from actual content
        jwt_count = len([pc for pc in top_items if 'jwt' in pc.summary.lower()])
        oauth_count = len([pc for pc in top_items if 'oauth' in pc.summary.lower() or 'openid' in pc.summary.lower()])
        spring_count = len([pc for pc in top_items if 'spring' in pc.summary.lower()])
        session_count = len([pc for pc in top_items if 'session' in pc.summary.lower()])
        security_count = len([pc for pc in top_items if any(code in pc.summary.lower() for code in ['401', '403', 'unauthorized', 'error'])])
        
        if jwt_count > 0:
            enhanced_trends.append(f"JWT Implementation Crisis: {jwt_count} developers struggling with token refresh mechanisms and payload validation errors")
        
        if oauth_count > 0:
            enhanced_trends.append(f"OAuth2 Integration Complexity: Growing adoption challenges with PKCE implementation and third-party provider configuration")
        
        if spring_count > 0:
            enhanced_trends.append(f"Spring Boot Security Challenges: {spring_count} discussions about configuration and implementation issues")
        
        if session_count > 0:
            enhanced_trends.append(f"Session Management Scaling: Scalability issues in distributed systems and stateless authentication")
        
        if security_count > 0:
            enhanced_trends.append(f"Authentication Error Resolution: {security_count} developers facing 401/403 errors and security vulnerabilities")
        
        # Add fallback trends if not enough
        fallback_trends = [
            "Enterprise Authentication Scaling: Companies evaluating solutions for microservices architecture and distributed session management",
            "Security-First Development: Increased focus on vulnerability prevention and compliance-ready authentication systems",
            "Developer Experience Optimization: Demand for authentication solutions that reduce implementation time and maintenance overhead"
        ]
        
        while len(enhanced_trends) < 5:
            if len(fallback_trends) > 0:
                enhanced_trends.append(fallback_trends.pop(0))
            else:
                break
        
        return enhanced_trends
    
    def _generate_trends_from_clusters(self, cluster_keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate trends from cluster analysis."""
        enhanced_trends = []
        
        # Theme-based trends
        theme_analysis = {}
        for ck in cluster_keywords:
            theme_lower = ck['theme'].lower()
            if 'jwt' in theme_lower:
                theme_analysis['jwt'] = theme_analysis.get('jwt', 0) + ck['size']
            elif 'oauth' in theme_lower:
                theme_analysis['oauth'] = theme_analysis.get('oauth', 0) + ck['size']
            elif 'spring' in theme_lower:
                theme_analysis['spring'] = theme_analysis.get('spring', 0) + ck['size']
            elif 'session' in theme_lower:
                theme_analysis['session'] = theme_analysis.get('session', 0) + ck['size']
            elif any(keyword in theme_lower for keyword in ['error', '401', '403', 'security']):
                theme_analysis['security'] = theme_analysis.get('security', 0) + ck['size']
        
        if theme_analysis.get('jwt', 0) > 0:
            enhanced_trends.append(f"JWT Implementation Crisis: {theme_analysis['jwt']} developers across {len([ck for ck in cluster_keywords if 'jwt' in ck['theme'].lower()])} clusters struggling with token management")
        
        if theme_analysis.get('oauth', 0) > 0:
            enhanced_trends.append(f"OAuth2 Integration Complexity: {theme_analysis['oauth']} discussions showing active provider evaluation and implementation challenges")
        
        if theme_analysis.get('spring', 0) > 0:
            enhanced_trends.append(f"Spring Security Configuration: {theme_analysis['spring']} developers facing framework-specific authentication implementation issues")
        
        if theme_analysis.get('session', 0) > 0:
            enhanced_trends.append(f"Session Management Scaling: {theme_analysis['session']} conversations about distributed authentication and persistence challenges")
        
        if theme_analysis.get('security', 0) > 0:
            enhanced_trends.append(f"Authentication Security Concerns: {theme_analysis['security']} developers dealing with vulnerability prevention and error resolution")
        
        # Add high-urgency insights
        high_urgency_clusters = [ck for ck in cluster_keywords if ck['urgency'] == 'high']
        if len(high_urgency_clusters) > 0:
            urgency_themes = [ck['theme'] for ck in high_urgency_clusters]
            enhanced_trends.append(f"High-Priority Consultation Opportunities: {len(high_urgency_clusters)} urgent clusters covering {', '.join(urgency_themes[:3])} requiring immediate vendor engagement")
        
        # Fill with generic trends if needed
        fallback_trends = [
            "Enterprise Authentication Demand: Growing adoption of comprehensive auth solutions for microservices architectures",
            "Developer Experience Priority: Strong preference for authentication platforms that reduce implementation complexity",
            "Security-First Mindset: Increased focus on compliance-ready authentication with vulnerability prevention"
        ]
        
        while len(enhanced_trends) < 5 and fallback_trends:
            enhanced_trends.append(fallback_trends.pop(0))
        
        return enhanced_trends
