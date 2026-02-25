from typing import Tuple
from constants import PLANT_DISEASES

class HierarchicalPlantDiseaseDetector:
    """
    Complete hierarchical pipeline for plant disease detection.
    
    Pipeline:
    1. Load and preprocess image
    2. Classify plant species using stage 1 model
    3. Load appropriate disease classifier
    4. Classify disease
    5. Return results with confidence scores
    """
    
    def __init__(
        self, 
        plant_classifier_path: str,
        disease_models_path: str,
        img_size: Tuple[int, int] = (224, 224),
        verbose: bool = True
    ):
        """
        Initialize the hierarchical detector.
        
        Args:
            plant_classifier_path: Path to the stage 1 plant classifier model
            disease_models_path: Directory containing disease classifier models
            img_size: Input image size (height, width)
            verbose: Whether to print loading messages
        """