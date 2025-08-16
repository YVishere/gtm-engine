"""Clustering utilities for content analysis."""

import numpy as np
import logging
from typing import List, Tuple
from collections import defaultdict
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

from models import ScrapedContent
from cluster_info import ClusterInfo
from config import Config


class ClusteringEngine:
    """Handles content clustering using embeddings."""
    
    def __init__(self, embedding_model, business_categorizer):
        self.embedding_model = embedding_model
        self.business_categorizer = business_categorizer
        self.logger = logging.getLogger(self.__class__.__name__)
        
        config = Config()
        
        # Clustering configuration
        self.clusterer = HDBSCAN(
            min_cluster_size=config.MIN_CLUSTER_SIZE,
            min_samples=2,
            metric='euclidean'
        )
        
        # UMAP for dimensionality reduction
        self.reducer = umap.UMAP(
            n_components=config.UMAP_COMPONENTS,
            random_state=42,
            metric='cosine'
        )
    
    def cluster_contents(self, embeddings: np.ndarray, contents: List[ScrapedContent]) -> List[ClusterInfo]:
        """Cluster content based on embeddings."""
        # Dimensionality reduction for better clustering
        reduced_embeddings = self.reducer.fit_transform(embeddings)
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(reduced_embeddings)
        
        # Organize clusters
        clusters_dict = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points
                clusters_dict[label].append((idx, contents[idx]))
        
        # Create ClusterInfo objects
        clusters = []
        for cluster_id, items in clusters_dict.items():
            cluster_contents = [item[1] for item in items]
            cluster_indices = [item[0] for item in items]
            
            # Find representative item (closest to cluster centroid)
            cluster_embeddings = embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = [cosine_similarity([emb], [centroid])[0][0] for emb in cluster_embeddings]
            rep_idx = cluster_indices[np.argmax(distances)]
            representative = contents[rep_idx]
            
            # Extract cluster theme
            cluster_texts = [f"{c.title} {c.content[:200]}" for c in cluster_contents]
            theme = self.extract_cluster_theme(cluster_texts)
            
            # Categorize cluster
            category = self.business_categorizer.categorize_technical_cluster(
                theme.split(), cluster_texts
            )
            
            # Determine urgency
            urgency = self.determine_cluster_urgency(cluster_contents, cluster_texts)
            
            cluster_info = ClusterInfo(
                id=cluster_id,
                items=cluster_contents,
                representative_item=representative,
                theme=theme,
                category=category,
                urgency_level=urgency,
                size=len(cluster_contents)
            )
            clusters.append(cluster_info)
        
        # Sort by importance (size * urgency)
        importance_scores = []
        for cluster in clusters:
            urgency_weight = {'high': 3, 'medium': 2, 'low': 1}[cluster.urgency_level]
            importance = cluster.size * urgency_weight
            importance_scores.append(importance)
        
        # Sort clusters by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        clusters = [clusters[i] for i in sorted_indices]
        
        print(f"   ✅ Created {len(clusters)} clusters (avg size: {np.mean([c.size for c in clusters]):.1f})")
        
        return clusters
    
    def extract_cluster_theme(self, cluster_texts: List[str]) -> str:
        """Extract main theme from cluster texts using TF-IDF with auth focus."""
        if len(cluster_texts) < 2:
            return "authentication implementation challenges"
        
        try:
            # Create combined text for analysis
            combined_text = ' '.join(cluster_texts).lower()
            
            # Check for specific auth patterns first
            auth_patterns = {
                'jwt': ['jwt', 'token', 'payload', 'refresh', 'validation', 'malformed'],
                'oauth': ['oauth', 'openid', 'oidc', 'pkce', 'authorization'],
                'session': ['session', 'cookie', 'stateless', 'jsessionid', 'persistence'],
                'spring': ['spring', 'boot', 'security', 'gateway', 'authorization-server'],
                'security': ['401', '403', 'unauthorized', 'forbidden', 'vulnerability'],
                'implementation': ['how to', 'configure', 'setup', 'implement', 'error'],
                'enterprise': ['saml', 'sso', 'ldap', 'enterprise', 'microservices']
            }
            
            # Score each pattern
            pattern_scores = {}
            for pattern_name, keywords in auth_patterns.items():
                score = sum(1 for keyword in keywords if keyword in combined_text)
                if score > 0:
                    pattern_scores[pattern_name] = score
            
            # Get top patterns
            if pattern_scores:
                top_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                
                # Create meaningful theme descriptions
                theme_descriptions = {
                    'jwt': 'JWT payload malformation and validation issues',
                    'oauth': 'OAuth2 and OpenID Connect integration challenges',
                    'session': 'Session management and stateless authentication',
                    'spring': 'Spring Boot security configuration problems',
                    'security': 'Authentication security vulnerabilities and errors',
                    'implementation': 'Authentication implementation and setup questions',
                    'enterprise': 'Enterprise authentication and SSO integration'
                }
                
                primary_theme = top_patterns[0][0]
                return theme_descriptions.get(primary_theme, f"{primary_theme} authentication issues")
            
            # Fallback to TF-IDF if no patterns match
            vectorizer = TfidfVectorizer(
                max_features=5,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[::-1][:2]
            
            top_terms = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            if top_terms:
                return f"{' '.join(top_terms)} authentication challenges"
            
            return "general authentication implementation issues"
        
        except Exception as e:
            self.logger.warning(f"Theme extraction failed: {e}")
            return "authentication discussion"
    
    def determine_cluster_urgency(self, cluster_contents: List[ScrapedContent], cluster_texts: List[str]) -> str:
        """Determine urgency level for a cluster."""
        urgency_keywords = {
            'high': ['urgent', 'broken', 'not working', 'error', 'fail', 'issue', 'problem', 'help'],
            'medium': ['how to', 'implement', 'setup', 'configure', 'question'],
            'low': ['discussion', 'general', 'opinion', 'comparison']
        }
        
        combined_text = ' '.join(cluster_texts).lower()
        
        urgency_scores = {}
        for level, keywords in urgency_keywords.items():
            score = sum(1 for kw in keywords if kw in combined_text)
            urgency_scores[level] = score
        
        # Also consider recency and engagement
        recent_count = sum(1 for c in cluster_contents if c.score > 5)  # High engagement
        if recent_count > len(cluster_contents) * 0.5:
            urgency_scores['high'] += 2
        
        return max(urgency_scores, key=urgency_scores.get)
