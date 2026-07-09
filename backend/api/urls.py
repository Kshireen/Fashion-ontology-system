from django.urls import path
from .views import (
    OntologyView,
    FeatureExtractionView,
    BatchProcessView,
    CSVUploadView,
    SearchView,
    ConceptPathView
)

urlpatterns = [
    path('api/ontology/', OntologyView.as_view(), name='ontology'),
    path('api/extract/', FeatureExtractionView.as_view(), name='extract'),
    path('api/batch/', BatchProcessView.as_view(), name='batch'),
    path('api/upload-csv/', CSVUploadView.as_view(), name='upload-csv'),
    path('api/search/', SearchView.as_view(), name='search'),
    path('api/concept-path/', ConceptPathView.as_view(), name='concept-path'),
]