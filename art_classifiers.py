#!/usr/bin/env python3
"""
Art Classification Models for MandrAIk Testing
==============================================

This module provides specialized art classification models for testing
the effectiveness of image protection methods on art images.
"""

import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import timm
from tensorflow.keras.applications import InceptionV3, ResNet50, EfficientNetB0
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


class ArtClassifier:
    """Base class for art classification models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.class_names = []
        self.input_size = (224, 224)
        
    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict image class with confidence scores."""
        raise NotImplementedError
        
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for the model."""
        raise NotImplementedError


class ArtBenchClassifier(ArtClassifier):
    """ArtBench-style classifier using pre-trained models fine-tuned on art datasets."""
    
    def __init__(self, model_name: str = 'artbench_resnet50'):
        super().__init__(model_name)
        
        # ArtBench-10 style classes
        self.art_styles = [
            'Baroque', 'Rococo', 'Neoclassicism', 'Romanticism', 
            'Impressionism', 'Post-Impressionism', 'Expressionism',
            'Cubism', 'Abstract Expressionism', 'Pop Art'
        ]
        
        # Use a pre-trained model and adapt it for art classification
        if 'resnet50' in model_name:
            self.model = timm.create_model('resnet50', pretrained=True, num_classes=len(self.art_styles))
            self.input_size = (224, 224)
        elif 'efficientnet' in model_name:
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(self.art_styles))
            self.input_size = (224, 224)
        elif 'vit' in model_name:
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(self.art_styles))
            self.input_size = (224, 224)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.eval()
        self.class_names = self.art_styles
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for PyTorch model."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict art style with confidence scores."""
        try:
            # Load and preprocess image
            image_tensor = self._load_and_preprocess_image(image_path)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Convert to list of (class_name, confidence) tuples
            results = []
            for i in range(top_k):
                class_name = self.class_names[top_indices[0][i].item()]
                confidence = top_probs[0][i].item()
                results.append((class_name, confidence))
            
            return results
            
        except Exception as e:
            print(f"Error predicting with ArtBench classifier: {e}")
            return [("unknown", 0.0)]


class WikiArtClassifier(ArtClassifier):
    """WikiArt-style classifier for art style classification."""
    
    def __init__(self, model_name: str = 'wikart_resnet50'):
        super().__init__(model_name)
        
        # WikiArt style classes (25 major art styles)
        self.art_styles = [
            'Abstract Expressionism', 'Abstract', 'Art Nouveau', 'Baroque',
            'Color Field Painting', 'Cubism', 'Early Renaissance', 'Expressionism',
            'Fauvism', 'High Renaissance', 'Impressionism', 'Mannerism',
            'Minimalism', 'Naive Art', 'Neoclassicism', 'Northern Renaissance',
            'Pop Art', 'Post-Impressionism', 'Realism', 'Rococo',
            'Romanticism', 'Symbolism', 'Synthetic Cubism', 'Ukiyo-e'
        ]
        
        # Use a pre-trained model adapted for WikiArt styles
        if 'resnet50' in model_name:
            self.model = timm.create_model('resnet50', pretrained=True, num_classes=len(self.art_styles))
            self.input_size = (224, 224)
        elif 'efficientnet' in model_name:
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(self.art_styles))
            self.input_size = (224, 224)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.eval()
        self.class_names = self.art_styles
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for PyTorch model."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict art style with confidence scores."""
        try:
            # Load and preprocess image
            image_tensor = self._load_and_preprocess_image(image_path)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Convert to list of (class_name, confidence) tuples
            results = []
            for i in range(top_k):
                class_name = self.class_names[top_indices[0][i].item()]
                confidence = top_probs[0][i].item()
                results.append((class_name, confidence))
            
            return results
            
        except Exception as e:
            print(f"Error predicting with WikiArt classifier: {e}")
            return [("unknown", 0.0)]


class NationalGalleryClassifier(ArtClassifier):
    """Classifier specifically for National Gallery of Art dataset categories."""
    
    def __init__(self, model_name: str = 'nga_resnet50'):
        super().__init__(model_name)
        
        # National Gallery of Art categories (from your dataset)
        self.art_categories = [
            'Graphite On Paper', 'Engraving On Laid Paper', 'Etching On Laid Paper',
            'Inkjet Print', 'Albumen Print', 'Drypoint', 'Portfolio', 'Painting',
            'Gelatin Silver Print', 'Engraving', 'Etching', 'Lithograph',
            'Watercolor', 'Oil On Canvas', 'Drawing', 'Print'
        ]
        
        # Use a pre-trained model adapted for NGA categories
        if 'resnet50' in model_name:
            self.model = timm.create_model('resnet50', pretrained=True, num_classes=len(self.art_categories))
            self.input_size = (224, 224)
        elif 'efficientnet' in model_name:
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(self.art_categories))
            self.input_size = (224, 224)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.eval()
        self.class_names = self.art_categories
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for PyTorch model."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict art category with confidence scores."""
        try:
            # Load and preprocess image
            image_tensor = self._load_and_preprocess_image(image_path)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Convert to list of (class_name, confidence) tuples
            results = []
            for i in range(top_k):
                class_name = self.class_names[top_indices[0][i].item()]
                confidence = top_probs[0][i].item()
                results.append((class_name, confidence))
            
            return results
            
        except Exception as e:
            print(f"Error predicting with NGA classifier: {e}")
            return [("unknown", 0.0)]


class ModernImageClassifier:
    """Wrapper for modern image classifiers (existing implementation)."""
    
    def __init__(self, model_name: str = 'inception_v3'):
        self.model_name = model_name
        
        # Load pre-trained model
        if model_name == 'inception_v3':
            self.model = InceptionV3(weights='imagenet', include_top=True)
            self.input_size = (299, 299)
            self.preprocess = inception_preprocess
        elif model_name == 'resnet50':
            self.model = ResNet50(weights='imagenet', include_top=True)
            self.input_size = (224, 224)
            self.preprocess = resnet_preprocess
        elif model_name == 'efficientnet_b0':
            self.model = EfficientNetB0(weights='imagenet', include_top=True)
            self.input_size = (224, 224)
            self.preprocess = efficientnet_preprocess
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get class names
        self.class_names = self._get_imagenet_class_names()
    
    def _get_imagenet_class_names(self) -> List[str]:
        """Get ImageNet class names."""
        # Create a simple test image to get predictions
        test_img = tf.random.normal((1, 299, 299, 3))
        
        # Use InceptionV3 for class names (most comprehensive)
        temp_model = InceptionV3(weights='imagenet', include_top=True)
        
        # Get predictions to extract class names
        preds = temp_model.predict(test_img, verbose=0)
        decoded = tf.keras.applications.inception_v3.decode_predictions(preds, top=1000)[0]
        
        # Extract class names
        class_names = [class_name for _, class_name, _ in decoded]
        
        return class_names
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict image class with confidence scores."""
        try:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.input_size)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = self.preprocess(x)
            
            # Get predictions
            preds = self.model.predict(x, verbose=0)
            
            # Decode predictions
            if self.model_name == 'inception_v3':
                decoded = tf.keras.applications.inception_v3.decode_predictions(preds, top=top_k)[0]
            elif self.model_name == 'resnet50':
                decoded = tf.keras.applications.resnet50.decode_predictions(preds, top=top_k)[0]
            elif self.model_name == 'efficientnet_b0':
                # EfficientNet doesn't have built-in decode_predictions, so we'll use the class names
                top_indices = np.argsort(preds[0])[-top_k:][::-1]
                decoded = [(i, self.class_names[i], float(preds[0][i])) for i in top_indices]
            
            # Return as list of (class_name, confidence) tuples
            return [(class_name, float(confidence)) for _, class_name, confidence in decoded]
            
        except Exception as e:
            print(f"Error predicting with {self.model_name}: {e}")
            return [("unknown", 0.0)]


class ArtClassifierFactory:
    """Factory class to create different types of art classifiers."""
    
    @staticmethod
    def create_classifier(classifier_type: str, model_name: str = None) -> ArtClassifier:
        """Create an art classifier based on type."""
        
        if classifier_type == 'artbench':
            return ArtBenchClassifier(model_name or 'artbench_resnet50')
        elif classifier_type == 'wikart':
            return WikiArtClassifier(model_name or 'wikart_resnet50')
        elif classifier_type == 'nga':
            return NationalGalleryClassifier(model_name or 'nga_resnet50')
        elif classifier_type in ['inception_v3', 'resnet50', 'efficientnet_b0']:
            return ModernImageClassifier(classifier_type)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    @staticmethod
    def get_available_classifiers() -> Dict[str, List[str]]:
        """Get list of available classifier types and their models."""
        return {
            'artbench': ['artbench_resnet50', 'artbench_efficientnet', 'artbench_vit'],
            'wikart': ['wikart_resnet50', 'wikart_efficientnet'],
            'nga': ['nga_resnet50', 'nga_efficientnet'],
            'modern': ['inception_v3', 'resnet50', 'efficientnet_b0']
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the art classifiers
    test_image = "test_images/test.jpg"  # Replace with actual test image
    
    if os.path.exists(test_image):
        print("Testing Art Classifiers...")
        
        # Test different classifier types
        classifiers = {
            'ArtBench': ArtClassifierFactory.create_classifier('artbench'),
            'WikiArt': ArtClassifierFactory.create_classifier('wikart'),
            'NGA': ArtClassifierFactory.create_classifier('nga'),
            'InceptionV3': ArtClassifierFactory.create_classifier('inception_v3')
        }
        
        for name, classifier in classifiers.items():
            print(f"\n{name} Classifier Results:")
            try:
                predictions = classifier.predict(test_image, top_k=3)
                for class_name, confidence in predictions:
                    print(f"  {class_name}: {confidence:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\nAvailable classifier types:")
    available = ArtClassifierFactory.get_available_classifiers()
    for classifier_type, models in available.items():
        print(f"  {classifier_type}: {', '.join(models)}") 