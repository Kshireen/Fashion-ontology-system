# core/scalable_processor.py
"""
Scalable Processing System for 100K+ Products
Handles large-scale batch processing with optimization
Supports 10+ categories with category-specific processing
"""

from unittest import result

from unittest import result

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue
import time
import logging
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Track processing statistics"""
    total_products: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Per-category stats
    by_category: Dict[str, Dict] = field(default_factory=lambda: defaultdict(dict))
    
    # Feature extraction stats
    features_extracted: int = 0
    avg_features_per_product: float = 0.0
    
    # Performance metrics
    processing_rate: float = 0.0  # products per second
    
    def calculate_metrics(self):
        """Calculate derived metrics"""
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            if duration > 0:
                self.processing_rate = self.processed / duration
        
        if self.processed > 0:
            self.avg_features_per_product = self.features_extracted / self.processed


@dataclass
class CategoryConfig:
    """Configuration for category-specific processing"""
    category_name: str
    category_id: str
    
    # Processing priorities
    enable_visual: bool = True
    enable_textual: bool = True
    
    # Category-specific features to extract
    priority_features: List[str] = field(default_factory=list)
    
    # Category-specific thresholds
    min_confidence: float = 0.5
    
    # Ontology mapping hints
    parent_category: Optional[str] = None


class BatchProcessor:
    """
    Efficient batch processor for large-scale product processing
    Uses chunking and parallel processing
    """
    
    def __init__(self, chunk_size: int = 1000, n_workers: int = 4):
        self.chunk_size = chunk_size
        self.n_workers = n_workers
        self.stats = ProcessingStats()
        
    def read_csv_in_chunks(self, filepath: str) -> Iterator[pd.DataFrame]:
        """Read CSV file in chunks for memory efficiency"""
        logger.info(f"Reading CSV from {filepath} in chunks of {self.chunk_size}")
        
        try:
            for chunk in pd.read_csv(filepath, chunksize=self.chunk_size):
                yield chunk
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
    
    def process_chunk(self, chunk: pd.DataFrame, extractor, 
                     category_configs: Dict[str, CategoryConfig]) -> List[Dict]:
        """Process a single chunk of products"""
        results = []
        
        for idx, row in chunk.iterrows():
            try:
                # Get category config
                category_id = str(row.get('category_id', ''))
                config = category_configs.get(category_id)

                desc = row.get('description')
                if pd.isna(desc) or str(desc).strip() == '' or str(desc).strip().lower() == 'nan':
                    desc = row.get('meta_info', '')
                if pd.isna(desc):
                    desc = ''
                
                # Prepare product data
                product_data = {
                    'product_id': str(row.get('product_id', f'P{idx}')),
                    'product_name': str(row.get('product_name', '')),
                    'description': str(desc or row.get('meta_info', '')),
                    'category_id': category_id,
                    'category_name': str(row.get('category_name', '')),
                    'department_id': str(row.get('department_id', '')),
                    'brand': str(row.get('brand', '')),
                    'feature_image': str(row.get('feature_image', '')),
                    'meta_info': str(row.get('meta_info', ''))
                }
                print(product_data["product_name"])
                print(product_data["description"])
                
                # Process based on category config
                if config:
                    result = self._process_with_config(
                        product_data, extractor, config
                    )
                else:
                    result = extractor.extract_features(product_data)
                
                feature_count = 0

                if hasattr(result, "features"):
                    feature_count = len(result.features)
                    self.stats.features_extracted += feature_count

                results.append({
                    "product_id": product_data["product_id"],
                    "category": product_data["category_name"],
                    "features": result.features if hasattr(result, "features") else result,
                    "success": True,
                })

                self.stats.processed += 1
                
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                self.stats.failed += 1
                results.append({
                    'product_id': product_data.get('product_id', f'P{idx}'),
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def _process_with_config(self, product_data: Dict, extractor, 
                            config: CategoryConfig) -> Dict:
        """Process product with category-specific configuration"""
        # Extract features with category hints
        result = extractor.extract_features(product_data)
        
        # Filter features based on category priority
        if config.priority_features:
            # Prioritize certain features for this category
            pass
        
        return result
    
    def process_parallel(self, chunks: List[pd.DataFrame], extractor,
                        category_configs: Dict[str, CategoryConfig]) -> List[Dict]:
        """Process multiple chunks in parallel"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for chunk in chunks:
                future = executor.submit(
                    self.process_chunk, chunk, extractor, category_configs
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
        
        return all_results


class CategoryManager:
    """
    Manages category-specific configurations and processing
    Handles 10+ fashion categories
    """
    
    def __init__(self):
        self.categories: Dict[str, CategoryConfig] = {}
        self._initialize_default_categories()
    
    def _initialize_default_categories(self):
        """Initialize default category configurations"""
        default_categories = [
            CategoryConfig(
                category_name="Dresses",
                category_id="dress",
                priority_features=["length", "neckline", "sleeve_length", "pattern"],
                parent_category="upper_body"
            ),
            CategoryConfig(
                category_name="Shirts",
                category_id="shirt",
                priority_features=["fit", "sleeve_length", "collar", "closure"],
                parent_category="upper_body"
            ),
            CategoryConfig(
                category_name="Pants",
                category_id="pants",
                priority_features=["fit", "length", "rise", "material"],
                parent_category="lower_body"
            ),
            CategoryConfig(
                category_name="Sneakers",
                category_id="sneakers",
                priority_features=["style", "closure", "material", "sole_type"],
                parent_category="footwear"
            ),
            CategoryConfig(
                category_name="Earrings",
                category_id="earrings",
                priority_features=["style", "material", "size", "closure_type"],
                parent_category="jewelry"
            ),
            CategoryConfig(
                category_name="Jackets",
                category_id="jackets",
                priority_features=["style", "length", "closure", "material"],
                parent_category="outerwear"
            ),
            CategoryConfig(
                category_name="Skirts",
                category_id="skirts",
                priority_features=["length", "fit", "waist_type", "pattern"],
                parent_category="lower_body"
            ),
            CategoryConfig(
                category_name="Bags",
                category_id="bags",
                priority_features=["style", "size", "material", "closure"],
                parent_category="accessories"
            ),
            CategoryConfig(
                category_name="Jeans",
                category_id="jeans",
                priority_features=["fit", "rise", "wash", "distressing"],
                parent_category="lower_body"
            ),
            CategoryConfig(
                category_name="T-Shirts",
                category_id="tshirts",
                priority_features=["fit", "neckline", "sleeve_length", "graphic"],
                parent_category="upper_body"
            )
        ]
        
        for config in default_categories:
            self.add_category(config)
    
    def add_category(self, config: CategoryConfig):
        """Add a category configuration"""
        self.categories[config.category_id] = config
        logger.info(f"Added category: {config.category_name}")
    
    def get_category(self, category_id: str) -> Optional[CategoryConfig]:
        """Get category configuration"""
        return self.categories.get(category_id)
    
    def get_all_categories(self) -> List[CategoryConfig]:
        """Get all category configurations"""
        return list(self.categories.values())


class ScalableProcessingPipeline:
    """
    Complete scalable processing pipeline for 100K+ products
    Integrates all components: extraction, learning, feedback
    """
    
    def __init__(self, extractor, learning_engine=None, 
                 chunk_size: int = 1000, n_workers: int = 4):
        self.extractor = extractor
        self.learning_engine = learning_engine
        
        self.batch_processor = BatchProcessor(chunk_size, n_workers)
        self.category_manager = CategoryManager()
        
        self.results_cache: Dict[str, Dict] = {}
        self.processing_history: List[Dict] = []
    
    def process_csv_file(self, filepath: str, 
                        max_products: Optional[int] = None) -> ProcessingStats:
        """
        Process entire CSV file
        Args:
            filepath: Path to CSV file
            max_products: Maximum products to process (None for all)
        """
        logger.info(f"Starting processing of {filepath}")
        
        self.batch_processor.stats = ProcessingStats()
        self.batch_processor.stats.start_time = datetime.now()
        
        all_results = []
        products_processed = 0
        
        # Get category configurations
        category_configs = {
            cat.category_id: cat 
            for cat in self.category_manager.get_all_categories()
        }
        
        # Process in chunks
        for chunk_idx, chunk in enumerate(
            self.batch_processor.read_csv_in_chunks(filepath)
        ):
            if max_products and products_processed >= max_products:
                logger.info(f"Reached maximum products limit: {max_products}")
                break
            
            # Limit chunk if necessary
            if max_products:
                remaining = max_products - products_processed
                chunk = chunk.head(remaining)
            
            logger.info(f"Processing chunk {chunk_idx + 1} ({len(chunk)} products)")
            
            # Process chunk
            chunk_results = self.batch_processor.process_chunk(
                chunk, self.extractor, category_configs
            )
            
            all_results.extend(chunk_results)
            products_processed += len(chunk)
            
            # Update stats
            self.batch_processor.stats.total_products = products_processed
            
            # Cache results
            for result in chunk_results:
                if result['success']:
                    self.results_cache[result['product_id']] = result
            
            # Log progress
            if (chunk_idx + 1) % 10 == 0:
                logger.info(
                    f"Progress: {products_processed} products processed, "
                    f"{self.batch_processor.stats.failed} failed"
                )
        
        # Finalize stats
        self.batch_processor.stats.end_time = datetime.now()
        self.batch_processor.stats.calculate_metrics()
        
        # Calculate per-category stats
        self._calculate_category_stats(all_results)
        
        # Save results
        self._save_results(all_results, filepath)
        
        logger.info("Processing complete!")
        self._log_final_stats()
        
        return self.batch_processor.stats
    
    def _calculate_category_stats(self, results: List[Dict]):
        """Calculate statistics per category"""
        category_counts = defaultdict(int)
        category_features = defaultdict(list)
        
        for result in results:
            if result['success']:
                category = result.get('category', 'unknown')
                category_counts[category] += 1
                
                features = result.get('features', {})
                if isinstance(features, dict):
                    category_features[category].append(len(features))
        
        # Update stats
        for category, count in category_counts.items():
            self.batch_processor.stats.by_category[category] = {
                'count': count,
                'avg_features': np.mean(category_features[category]) 
                               if category_features[category] else 0
            }
    
    def _save_results(self, results: List[Dict], original_filepath: str):
        """Save processing results"""
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"processed_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'source_file': original_filepath,
                    'processing_date': timestamp,
                    'total_products': len(results)
                },
                'results': results[:100]  # Save sample for review
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def _log_final_stats(self):
        """Log final processing statistics"""
        stats = self.batch_processor.stats
        
        logger.info("=" * 80)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total Products: {stats.total_products}")
        logger.info(f"Processed: {stats.processed}")
        logger.info(f"Failed: {stats.failed}")
        logger.info(f"Success Rate: {stats.processed / stats.total_products * 100:.2f}%")
        logger.info(f"Processing Rate: {stats.processing_rate:.2f} products/second")
        logger.info("")
        logger.info("Per-Category Breakdown:")
        for category, cat_stats in stats.by_category.items():
            logger.info(f"  {category}: {cat_stats['count']} products, "
                       f"avg {cat_stats['avg_features']:.1f} features")
        logger.info("=" * 80)
    
    def get_product_result(self, product_id: str) -> Optional[Dict]:
        """Retrieve result for specific product"""
        return self.results_cache.get(product_id)
    
    def search_by_category(self, category: str) -> List[Dict]:
        """Get all results for a specific category"""
        return [
            result for result in self.results_cache.values()
            if result.get('category') == category
        ]


# ============================================================================
# Usage Examples
# ============================================================================

def demo_scalable_processing():
    """Demonstrate scalable processing"""
    from core.ontology_engine import (
        LexicalLayer, ConceptLayer, InstanceLayer, MultimodalFeatureExtractor
    )
    
    logger.info("=" * 80)
    logger.info("SCALABLE PROCESSING DEMO (100K+ Products)")
    logger.info("=" * 80)
    
    # Initialize system
    lexical = LexicalLayer()
    concept = ConceptLayer()
    instance = InstanceLayer()
    extractor = MultimodalFeatureExtractor()
    
    # Initialize pipeline
    pipeline = ScalableProcessingPipeline(
        extractor=extractor,
        chunk_size=1000,
        n_workers=4
    )
    
    # Show category configurations
    logger.info("\nConfigured Categories:")
    for config in pipeline.category_manager.get_all_categories():
        logger.info(f"  - {config.category_name} (Priority: {config.priority_features})")
    
    logger.info("\nPipeline ready to process 100K+ products")
    logger.info("Usage: pipeline.process_csv_file('path/to/data.csv')")
    
    # Example processing (would process actual file)
    logger.info("\nExpected Performance:")
    logger.info("  - Processing Rate: ~100-500 products/second")
    logger.info("  - Memory Usage: ~2-4 GB (with chunking)")
    logger.info("  - Time for 100K products: ~3-15 minutes")
    

if __name__ == "__main__":
    demo_scalable_processing()