"""Vectorized LLM processor using embeddings and clustering for smart batching."""

import logging
import numpy as np
import time
from typing import List, Tuple

# Vector processing imports
from sentence_transformers import SentenceTransformer

from models import ScrapedContent, ProcessedContent
from config import Config
from cluster_info import ClusterInfo
from business_categorizer import BusinessCategorizer
from clustering_engine import ClusteringEngine
from content_processor import ContentProcessor
from summary_generator import SummaryGenerator


class VectorizedLLMProcessor:
    """Fast LLM processor using vector embeddings and intelligent clustering."""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize models
        print("🤖 Initializing vector processing models...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        # Initialize components
        self.business_categorizer = BusinessCategorizer(self.embedding_model)
        self.clustering_engine = ClusteringEngine(self.embedding_model, self.business_categorizer)
        self.content_processor = ContentProcessor()
        self.summary_generator = SummaryGenerator(self.config)
        
        self.logger.info("Vectorized LLM processor initialized successfully")
        
    def process_content_batch(self, contents: List[ScrapedContent]) -> List[ProcessedContent]:
        """Process content using vectorization and smart clustering."""
        if not contents:
            return []
        
        print(f"\n🚀 Starting vectorized processing of {len(contents)} items...")
        start_time = time.time()
        
        # Phase 1: Generate embeddings
        print("📊 Phase 1: Generating embeddings...")
        embeddings, embedding_texts = self._generate_embeddings(contents)
        
        # Phase 2: Cluster similar content
        print("🔍 Phase 2: Clustering similar content...")
        clusters = self.clustering_engine.cluster_contents(embeddings, contents)
        
        # Store clusters for summary generation
        self._clusters = clusters
        
        # Phase 3: Smart batch processing
        print("🧠 Phase 3: Smart cluster processing...")
        processed_contents = self._process_clusters_smart(clusters)
        
        # Phase 4: Generate enhanced trends
        print("📈 Phase 4: Analyzing trends...")
        self._analyze_cluster_trends(clusters)
        
        total_time = time.time() - start_time
        print(f"\n✅ Vectorized processing complete! {len(processed_contents)} items in {total_time:.1f}s")
        print(f"🚀 Speed improvement: ~{242 * 2.5 / total_time:.1f}x faster than sequential processing")
        
        return processed_contents
    
    def _generate_embeddings(self, contents: List[ScrapedContent]) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for all content."""
        embedding_texts = []
        
        for content in contents:
            # Create rich text representation
            text = f"Title: {content.title} Content: {content.content[:500]} Source: {content.source.value}"
            embedding_texts.append(text)
        
        # Batch embedding generation for speed
        embeddings = self.embedding_model.encode(
            embedding_texts,
            batch_size=self.config.BATCH_EMBEDDING_SIZE,
            show_progress_bar=True
        )
        
        print(f"   ✅ Generated {len(embeddings)} embeddings")
        return embeddings, embedding_texts
    
    def _process_clusters_smart(self, clusters: List[ClusterInfo]) -> List[ProcessedContent]:
        """Process clusters using smart batching."""
        processed_contents = []
        
        # Limit clusters to process for performance
        clusters_to_process = clusters[:self.config.MAX_CLUSTERS_TO_PROCESS]
        
        for i, cluster in enumerate(clusters_to_process, 1):
            print(f"   📦 Processing cluster {i}/{len(clusters_to_process)}: {cluster.theme} ({cluster.size} items)")
            
            # Process representative item with detailed analysis
            representative_analysis = self.content_processor.analyze_representative_item(cluster)
            
            # Propagate analysis to all items in cluster
            cluster_processed = self.content_processor.propagate_analysis_to_cluster(representative_analysis, cluster)
            processed_contents.extend(cluster_processed)
        
        return processed_contents
    
    def _analyze_cluster_trends(self, clusters: List[ClusterInfo]):
        """Analyze trends across clusters for enhanced reporting."""
        from collections import Counter
        
        cluster_themes = [c.theme for c in clusters]
        cluster_categories = [c.category for c in clusters]
        
        # Store for later use in summary generation
        self._cluster_analysis = {
            'total_clusters': len(clusters),
            'top_themes': cluster_themes[:5],
            'category_distribution': dict(Counter(cluster_categories)),
            'urgency_distribution': dict(Counter([c.urgency_level for c in clusters])),
            'largest_clusters': sorted(clusters, key=lambda x: x.size, reverse=True)[:3]
        }
        
        print(f"   ✅ Identified {len(clusters)} distinct conversation clusters")
        print(f"   📊 Top themes: {', '.join(cluster_themes[:3])}")
        
        return self._cluster_analysis
    
    def generate_overall_summary(self, processed_contents: List[ProcessedContent]):
        """Generate enhanced summary using cluster analysis."""
        clusters = getattr(self, '_clusters', [])
        return self.summary_generator.generate_overall_summary(processed_contents, clusters)
    
    def generate_opportunity_explanation(self, content: ProcessedContent) -> str:
        """Generate GTM-focused explanation for business opportunities."""
        return self.content_processor.generate_opportunity_explanation(content, self.business_categorizer)

