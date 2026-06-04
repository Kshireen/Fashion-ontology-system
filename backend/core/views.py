"""
Django REST API for Fashion Ontology System
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.core.files.storage import default_storage
import pandas as pd
import json
from typing import Dict, List

# Import our ontology engine
from core.ontology_engine import MultimodalFeatureExtractor, ProductInstance


# Global extractor instance (in production, use proper dependency injection)
extractor = MultimodalFeatureExtractor()


class OntologyView(APIView):
    """
    API endpoint for ontology operations
    GET: Retrieve ontology structure
    POST: Add new concepts to ontology
    """
    
    def get(self, request):
        """Get complete ontology structure"""
        try:
            # Get full ontology tree
            concept_tree = extractor._build_concept_tree()
            
            # Get lexical terms
            lexical_terms = []
            for canonical, term in extractor.lexical_layer.terms.items():
                lexical_terms.append({
                    "canonical": canonical,
                    "aliases": list(term.aliases),
                    "languages": list(term.languages)
                })
            
            response_data = {
                "success": True,
                "data": {
                    "concept_layer": concept_tree,
                    "lexical_layer": lexical_terms,
                    "statistics": {
                        "total_concepts": len(extractor.concept_layer.concepts),
                        "total_lexical_terms": len(extractor.lexical_layer.terms),
                        "total_instances": len(extractor.instance_layer.instances)
                    }
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FeatureExtractionView(APIView):
    """
    API endpoint for feature extraction
    POST: Extract features from product data
    """
    
    def post(self, request):
        """Extract features from product"""
        try:
            product_data = request.data
            
            # Validate required fields
            required_fields = ['product_id', 'product_name']
            for field in required_fields:
                if field not in product_data:
                    return Response({
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }, status=status.HTTP_400_BAD_REQUEST)
            
            # Extract features
            instance = extractor.extract_features(product_data)
            
            # Build response with full ontology paths
            feature_mappings = {}
            for feature_type, concept_id in instance.features.items():
                path = extractor.concept_layer.get_concept_path(concept_id)
                feature_mappings[feature_type] = {
                    "concept_id": concept_id,
                    "path": " > ".join(path)
                }
            
            response_data = {
                "success": True,
                "data": {
                    "product_id": instance.id,
                    "product_name": instance.name,
                    "extracted_features": {
                        "lexical_terms": instance.raw_textual_features,
                        "ontology_mappings": feature_mappings
                    },
                    "metadata": {
                        "total_features_extracted": len(instance.features),
                        "brand": instance.brand,
                        "category": instance.category_id
                    }
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BatchProcessView(APIView):
    """
    API endpoint for batch processing
    POST: Process multiple products at once
    """
    
    def post(self, request):
        """Process multiple products"""
        try:
            products = request.data.get('products', [])
            
            if not products:
                return Response({
                    "success": False,
                    "error": "No products provided"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            results = []
            errors = []
            
            for idx, product_data in enumerate(products):
                try:
                    instance = extractor.extract_features(product_data)
                    
                    feature_mappings = {}
                    for feature_type, concept_id in instance.features.items():
                        path = extractor.concept_layer.get_concept_path(concept_id)
                        feature_mappings[feature_type] = " > ".join(path)
                    
                    results.append({
                        "product_id": instance.id,
                        "product_name": instance.name,
                        "features_extracted": len(instance.features),
                        "ontology_mappings": feature_mappings
                    })
                    
                except Exception as e:
                    errors.append({
                        "index": idx,
                        "product_id": product_data.get('product_id', 'unknown'),
                        "error": str(e)
                    })
            
            response_data = {
                "success": True,
                "data": {
                    "processed": len(results),
                    "failed": len(errors),
                    "results": results,
                    "errors": errors
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CSVUploadView(APIView):
    """
    API endpoint for CSV file upload and processing
    POST: Upload CSV file with product data
    """
    
    def post(self, request):
        """Process CSV file"""
        try:
            if 'file' not in request.FILES:
                return Response({
                    "success": False,
                    "error": "No file provided"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            csv_file = request.FILES['file']
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Process only first 100 rows for POC
            df = df.head(100)
            
            results = []
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    product_data = {
                        'product_id': str(row.get('product_id', f'P{idx}')),
                        'product_name': str(row.get('product_name', '')),
                        'description': str(row.get('description', '')),
                        'category_id': str(row.get('category_id', '')),
                        'department_id': str(row.get('department_id', '')),
                        'brand': str(row.get('brand', '')),
                        'feature_image': str(row.get('feature_image', ''))
                    }
                    
                    instance = extractor.extract_features(product_data)
                    
                    results.append({
                        "product_id": instance.id,
                        "product_name": instance.name,
                        "features_count": len(instance.features),
                        "lexical_terms": instance.raw_textual_features
                    })
                    
                except Exception as e:
                    errors.append({
                        "row": idx,
                        "error": str(e)
                    })
            
            response_data = {
                "success": True,
                "data": {
                    "total_rows": len(df),
                    "processed": len(results),
                    "failed": len(errors),
                    "results": results[:10],  # Return first 10 for preview
                    "errors": errors[:10]
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SearchView(APIView):
    """
    API endpoint for searching products by features
    POST: Search products using ontology concepts
    """
    
    def post(self, request):
        """Search products by features"""
        try:
            query = request.data.get('query', {})
            
            # Search in instance layer
            results = extractor.instance_layer.query_by_features(query)
            
            # Format results
            formatted_results = []
            for instance in results:
                feature_paths = {}
                for feature_type, concept_id in instance.features.items():
                    path = extractor.concept_layer.get_concept_path(concept_id)
                    feature_paths[feature_type] = " > ".join(path)
                
                formatted_results.append({
                    "product_id": instance.id,
                    "product_name": instance.name,
                    "brand": instance.brand,
                    "features": feature_paths
                })
            
            response_data = {
                "success": True,
                "data": {
                    "query": query,
                    "results_count": len(formatted_results),
                    "results": formatted_results
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ConceptPathView(APIView):
    """
    API endpoint for concept path operations
    GET: Get path for a concept in the ontology
    """
    
    def get(self, request):
        """Get concept path"""
        try:
            concept_id = request.query_params.get('concept_id')
            
            if not concept_id:
                return Response({
                    "success": False,
                    "error": "concept_id parameter required"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if concept_id not in extractor.concept_layer.concepts:
                return Response({
                    "success": False,
                    "error": f"Concept '{concept_id}' not found"
                }, status=status.HTTP_404_NOT_FOUND)
            
            path = extractor.concept_layer.get_concept_path(concept_id)
            concept = extractor.concept_layer.concepts[concept_id]
            
            response_data = {
                "success": True,
                "data": {
                    "concept_id": concept_id,
                    "concept_name": concept.name,
                    "concept_type": concept.type.value,
                    "path": path,
                    "path_string": " > ".join(path),
                    "children_count": len(concept.children)
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


