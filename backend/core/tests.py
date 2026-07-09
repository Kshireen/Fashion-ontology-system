from django.test import TestCase




from ontology_engine import MultimodalFeatureExtractor
from scalable_processor import ScalableProcessingPipeline

extractor = MultimodalFeatureExtractor()
pipeline = ScalableProcessingPipeline(extractor=extractor, chunk_size=50, n_workers=2)

filepath = r"D:\BOOKS\Machine Learning\FS\fashion-ontology-engine\data\raw\Shirts.csv"
stats = pipeline.process_csv_file(filepath, max_products=20)
print(stats)