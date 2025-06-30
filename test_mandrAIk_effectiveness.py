#!/usr/bin/env python3
"""
Enhanced MandrAIk Effectiveness Test Suite
==========================================

This module provides comprehensive testing of MandrAIk protection methods
with support for semantic targets, dataset testing, and art-specific classifiers.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import cv2
from scipy.stats import entropy

# Import MandrAIk
from MandrAIk import MandrAIk

# Import art classifiers
try:
    from art_classifiers import (
        ArtClassifierFactory, 
        ArtBenchClassifier, 
        WikiArtClassifier, 
        NationalGalleryClassifier,
        ModernImageClassifier
    )
    ART_CLASSIFIERS_AVAILABLE = True
except ImportError:
    print("Warning: Art classifiers not available. Using only standard models.")
    ART_CLASSIFIERS_AVAILABLE = False


class ModernImageClassifier:
    """Wrapper for modern image classifiers (fallback if art_classifiers not available)."""
    
    def __init__(self, model_name: str = 'inception_v3'):
        import tensorflow as tf
        from tensorflow.keras.applications import InceptionV3, ResNet50, EfficientNetB0
        from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
        from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
        
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
        import tensorflow as tf
        from tensorflow.keras.applications import InceptionV3
        
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
        import tensorflow as tf
        from tensorflow.keras.applications.inception_v3 import decode_predictions as inception_decode
        from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode
        
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
                decoded = inception_decode(preds, top=top_k)[0]
            elif self.model_name == 'resnet50':
                decoded = resnet_decode(preds, top=top_k)[0]
            elif self.model_name == 'efficientnet_b0':
                # EfficientNet doesn't have built-in decode_predictions, so we'll use the class names
                top_indices = np.argsort(preds[0])[-top_k:][::-1]
                decoded = [(i, self.class_names[i], float(preds[0][i])) for i in top_indices]
            
            # Return as list of (class_name, confidence) tuples
            return [(class_name, float(confidence)) for _, class_name, confidence in decoded]
            
        except Exception as e:
            print(f"Error predicting with {self.model_name}: {e}")
            return [("unknown", 0.0)]


class MandrAIkEffectivenessTester:
    """Enhanced effectiveness tester for MandrAIk with semantic targets, dataset support, and art classifiers."""
    
    def __init__(self, test_images_dir: str = "test_images", output_dir: str = "test_results", 
                 target_images_dir: str = "target_images", test_dataset_dir: str = None):
        """Initialize the tester with semantic target, dataset support, and art classifiers."""
        self.test_images_dir = Path(test_images_dir) if test_images_dir else None
        self.test_dataset_dir = Path(test_dataset_dir) if test_dataset_dir else None
        self.output_dir = Path(output_dir)
        self.target_images_dir = Path(target_images_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Define strength values for different protection levels
        self.strength_values = {
            'low': 0.05,
            'medium': 0.1,
            'high': 0.2
        }
        
        # Load models - focus on standard models that actually work
        self.models = {}
        
        # Standard models (these work well and give meaningful predictions)
        self.models.update({
            'inception_v3': ModernImageClassifier('inception_v3'),
            'resnet50': ModernImageClassifier('resnet50'),
            'efficientnet_b0': ModernImageClassifier('efficientnet_b0')
        })
        
        # Add pre-trained art classifier from Hugging Face
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            from PIL import Image
            
            class HuggingFaceArtClassifier:
                """Pre-trained art classifier from Hugging Face."""
                
                def __init__(self, model_name: str = 'oschamp/vit-artworkclassifier'):
                    self.processor = AutoImageProcessor.from_pretrained(model_name)
                    self.model = AutoModelForImageClassification.from_pretrained(model_name)
                    self.model.eval()
                    self.class_names = list(self.model.config.id2label.values())
                    self.model_name = model_name
                
                def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
                    """Predict art style with confidence scores."""
                    try:
                        # Load and preprocess image
                        image = Image.open(image_path).convert('RGB')
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        # Get predictions
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            probabilities = torch.softmax(outputs.logits, dim=1)
                            top_probs, top_indices = torch.topk(probabilities, top_k)
                        
                        # Convert to list of (class_name, confidence) tuples
                        results = []
                        for i in range(top_k):
                            class_name = self.class_names[top_indices[0][i].item()]
                            confidence = top_probs[0][i].item()
                            results.append((class_name, confidence))
                        
                        return results
                        
                    except Exception as e:
                        print(f"Error predicting with {self.model_name}: {e}")
                        return [("unknown", 0.0)]
            
            # Add the pre-trained art classifier
            self.models['huggingface_art'] = HuggingFaceArtClassifier()
            print("✓ Pre-trained art classifier loaded: oschamp/vit-artworkclassifier")
            print(f"  Art styles: {self.models['huggingface_art'].class_names}")
            
        except Exception as e:
            print(f"⚠ Could not load pre-trained art classifier: {e}")
        
        print("✓ Standard ImageNet classifiers loaded (inception_v3, resnet50, efficientnet_b0)")
        
        # Load target images for semantic methods
        self.target_images = self._load_target_images()
        print(f"Loaded {len(self.target_images)} target images: {[img.name for img in self.target_images]}")
        
        # Define art categories (from National Gallery of Art dataset)
        self.art_categories = [
            'Graphite On Paper',
            'Engraving On Laid Paper', 
            'Etching On Laid Paper',
            'Inkjet Print',
            'Albumen Print',
            'Drypoint',
            'Portfolio',
            'Painting',
            'Engraving',
            'Etching',
            'Lithograph',
            'Watercolor',
            'Oil On Canvas',
            'Drawing',
            'Print'
        ]
        
        # Results storage
        self.results = {
            'summary': {
                'total_tests': 0,
                'successful_protections': 0,
                'protection_success_rate': 0.0,
                'layer_used': 'mixed7',
                'timestamp': datetime.now().isoformat()
            },
            'by_model': {},
            'by_method': {},
            'by_strength': {},
            'by_noise_type': {},
            'by_category': {},
            'detailed_results': []
        }
    
    def _load_target_images(self) -> List[Path]:
        """Load target images for semantic methods."""
        target_images = []
        
        if self.target_images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                target_images.extend(self.target_images_dir.glob(ext))
                target_images.extend(self.target_images_dir.glob(ext.upper()))
        
        return target_images
    
    def _find_semantic_targets(self, image_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """Find semantic target images for the given image."""
        if len(self.target_images) < 2:
            return None, None
        
        # Simple strategy: use first two target images
        return self.target_images[0], self.target_images[1]
    
    def _calculate_confidence_entropy(self, predictions: List[Tuple[str, float]]) -> float:
        """Calculate entropy of confidence distribution."""
        confidences = [conf for _, conf in predictions]
        if sum(confidences) == 0:
            return 0.0
        # Normalize confidences
        confidences = np.array(confidences) / sum(confidences)
        return entropy(confidences)
    
    def _calculate_psnr(self, original_path: str, protected_path: str) -> float:
        """Calculate PSNR between original and protected images."""
        try:
            original = cv2.imread(original_path)
            protected = cv2.imread(protected_path)
            
            if original is None or protected is None:
                return 0.0
            
            # Ensure same size
            if original.shape != protected.shape:
                protected = cv2.resize(protected, (original.shape[1], original.shape[0]))
            
            # Calculate MSE
            mse = np.mean((original.astype(np.float64) - protected.astype(np.float64)) ** 2)
            
            if mse == 0:
                return float('inf')
            
            # Calculate PSNR
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            return psnr
            
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            return 0.0
    
    def _calculate_ssim(self, original_path: str, protected_path: str) -> float:
        """Calculate SSIM between original and protected images."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            original = cv2.imread(original_path)
            protected = cv2.imread(protected_path)
            
            if original is None or protected is None:
                return 0.0
            
            # Ensure same size
            if original.shape != protected.shape:
                protected = cv2.resize(protected, (original.shape[1], original.shape[0]))
            
            # Convert to grayscale for SSIM
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            protected_gray = cv2.cvtColor(protected, cv2.COLOR_BGR2GRAY)
            
            return ssim(original_gray, protected_gray)
            
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0.0
    
    def _calculate_mse(self, original_path: str, protected_path: str) -> float:
        """Calculate MSE between original and protected images."""
        try:
            original = cv2.imread(original_path)
            protected = cv2.imread(protected_path)
            
            if original is None or protected is None:
                return 0.0
            
            # Ensure same size
            if original.shape != protected.shape:
                protected = cv2.resize(protected, (original.shape[1], original.shape[0]))
            
            # Calculate MSE
            mse = np.mean((original.astype(np.float64) - protected.astype(np.float64)) ** 2)
            return mse
            
        except Exception as e:
            print(f"Error calculating MSE: {e}")
            return 0.0
    
    def test_single_image(self, image_path: Path, protection_strength: str = 'medium', method: str = 'poison', noise_type: str = 'perlin') -> Dict:
        """Test a single image with specified protection method, strength, and noise type."""
        try:
            # Find semantic targets for this image (skip for poison and fourier)
            if method not in ['poison', 'fourier']:
                target_image1, target_image2 = self._find_semantic_targets(image_path)
                if not target_image1 or not target_image2:
                    print(f"  No semantic targets found for {image_path.name}")
                    return {}
            else:
                target_image1, target_image2 = None, None
            
            # Set up output path
            output_path = self.output_dir / f"protected_{image_path.stem}_{method}_{noise_type}_{protection_strength}{image_path.suffix}"
            
            # Initialize MandrAIk
            mandrAIk = MandrAIk()
            
            # Apply protection based on method
            start_time = time.time()
            
            if method == 'poison':
                mandrAIk.poison(
                    image_path=str(image_path),
                    output_path=str(output_path),
                    protection_strength=self.strength_values[protection_strength],
                    noise_type=noise_type
                )
            elif method == 'fgsm_poison':
                mandrAIk.fgsm_poison(
                    image_path=str(image_path),
                    target_image_path1=str(target_image1),
                    target_image_path2=str(target_image2),
                    output_path=str(output_path)
                )
            elif method == 'fourier':
                mandrAIk.fourier(
                    image_path=str(image_path),
                    output_path=str(output_path),
                    protection_strength=self.strength_values[protection_strength],
                    frequency_band='high',
                    adaptive=True
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            protection_time = time.time() - start_time
            
            # Test with all models
            results = {
                'image_path': str(image_path),
                'target_image1': str(target_image1) if method not in ['poison', 'fourier'] else None,
                'target_image2': str(target_image2) if method not in ['poison', 'fourier'] else None,
                'method': method,
                'noise_type': noise_type,
                'protected_path': str(output_path),
                'protection_strength': protection_strength,
                'strength_value': self.strength_values[protection_strength],
                'protection_time': protection_time,
                'layer_used': 'mixed7',
                'models': {}
            }
            
            # Test original image
            original_results = {}
            for model_name, model in self.models.items():
                try:
                    original_preds = model.predict(str(image_path))
                    original_results[model_name] = {
                        'original_top1': original_preds[0],
                        'original_top5': original_preds[:5],
                        'original_confidence_entropy': self._calculate_confidence_entropy(original_preds)
                    }
                except Exception as e:
                    print(f"Error testing original image with {model_name}: {e}")
                    original_results[model_name] = None
            
            # Test protected image
            protected_results = {}
            for model_name, model in self.models.items():
                try:
                    protected_preds = model.predict(str(output_path))
                    protected_results[model_name] = {
                        'protected_top1': protected_preds[0],
                        'protected_top5': protected_preds[:5],
                        'protected_confidence_entropy': self._calculate_confidence_entropy(protected_preds)
                    }
                except Exception as e:
                    print(f"Error testing protected image with {model_name}: {e}")
                    protected_results[model_name] = None
            
            # Calculate metrics for each model
            for model_name in self.models.keys():
                if original_results.get(model_name) and protected_results.get(model_name):
                    orig_data = original_results[model_name]
                    prot_data = protected_results[model_name]
                    
                    # Calculate confidence reduction
                    orig_conf = orig_data['original_top1'][1]
                    prot_conf = prot_data['protected_top1'][1]
                    confidence_reduction = orig_conf - prot_conf
                    confidence_reduction_percentage = (confidence_reduction / orig_conf) * 100 if orig_conf > 0 else 0
                    
                    # Check if confidence was reduced
                    confidence_reduced = confidence_reduction > 0
                    
                    # Check if attack was successful (top-1 class changed)
                    attack_success = orig_data['original_top1'][0] != prot_data['protected_top1'][0]
                    
                    # Calculate entropy change
                    entropy_change = prot_data['protected_confidence_entropy'] - orig_data['original_confidence_entropy']
                    
                    # Calculate quality metrics
                    psnr = self._calculate_psnr(str(image_path), str(output_path))
                    ssim = self._calculate_ssim(str(image_path), str(output_path))
                    mse = self._calculate_mse(str(image_path), str(output_path))
                    
                    results['models'][model_name] = {
                        'original_top1': orig_data['original_top1'],
                        'protected_top1': prot_data['protected_top1'],
                        'original_top5': orig_data['original_top5'],
                        'protected_top5': prot_data['protected_top5'],
                        'confidence_reduction': confidence_reduction,
                        'confidence_reduction_percentage': confidence_reduction_percentage,
                        'confidence_reduced': confidence_reduced,
                        'attack_success': attack_success,
                        'original_confidence_entropy': orig_data['original_confidence_entropy'],
                        'protected_confidence_entropy': prot_data['protected_confidence_entropy'],
                        'entropy_increase': entropy_change,
                        'psnr': psnr,
                        'ssim': ssim,
                        'mse': mse
                    }
            
            # Determine category from image path
            category = self._determine_category(image_path)
            results['category'] = category
            
            return results
            
        except Exception as e:
            print(f"Error testing image {image_path}: {e}")
            return {}
    
    def _determine_category(self, image_path: Path) -> str:
        """Determine art category from image path."""
        if self.test_dataset_dir:
            # Extract category from path structure
            try:
                relative_path = image_path.relative_to(self.test_dataset_dir)
                category = relative_path.parts[0] if relative_path.parts else "Unknown"
                return category
            except:
                pass
        
        # Fallback: try to extract from filename
        filename = image_path.name.lower()
        for category in self.art_categories:
            if category.lower().replace(' ', '_') in filename:
                return category
        
        return "Unknown"
    
    def run_comprehensive_test(self, methods: List[str] = None, strengths: List[str] = None, 
                             noise_types: List[str] = None, images_per_category: int = 30) -> Dict:
        """Run comprehensive test with all specified parameters."""
        
        if methods is None:
            methods = ['poison', 'fourier']
        if strengths is None:
            strengths = ['medium']
        if noise_types is None:
            noise_types = ['perlin']
        
        print(f"Running comprehensive test with:")
        print(f"  Methods: {methods}")
        print(f"  Strengths: {strengths}")
        print(f"  Noise types: {noise_types}")
        print(f"  Images per category: {images_per_category}")
        print(f"  Models: {list(self.models.keys())}")
        
        # Get test images
        if self.test_dataset_dir:
            test_images = self._get_dataset_images(images_per_category)
        else:
            test_images = self._get_test_images()
        
        print(f"Found {len(test_images)} test images")
        
        # Run tests
        total_tests = 0
        successful_protections = 0
        
        for image_path in test_images:
            print(f"\nTesting {image_path.name}...")
            
            for method in methods:
                for strength in strengths:
                    for noise_type in noise_types:
                        try:
                            result = self.test_single_image(
                                image_path, strength, method, noise_type
                            )
                            
                            if result:
                                self.results['detailed_results'].append(result)
                                total_tests += 1
                                successful_protections += 1
                                
                                # Print progress
                                print(f"  ✓ {method} ({noise_type}, {strength}) - {len(result.get('models', {}))} models tested")
                            else:
                                print(f"  ✗ {method} ({noise_type}, {strength}) - failed")
                                
                        except Exception as e:
                            print(f"  ✗ {method} ({noise_type}, {strength}) - error: {e}")
        
        # Aggregate results
        self._aggregate_results()
        
        # Update summary
        self.results['summary']['total_tests'] = total_tests
        self.results['summary']['successful_protections'] = successful_protections
        self.results['summary']['protection_success_rate'] = successful_protections / total_tests if total_tests > 0 else 0
        
        return self.results
    
    def _get_dataset_images(self, images_per_category: int) -> List[Path]:
        """Get images from dataset directory."""
        images = []
        
        if not self.test_dataset_dir.exists():
            print(f"Dataset directory {self.test_dataset_dir} does not exist")
            return images
        
        # Get all category directories
        category_dirs = [d for d in self.test_dataset_dir.iterdir() if d.is_dir()]
        
        for category_dir in category_dirs:
            category_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                category_images.extend(category_dir.glob(ext))
                category_images.extend(category_dir.glob(ext.upper()))
            
            # Take up to images_per_category from each category
            selected_images = category_images[:images_per_category]
            images.extend(selected_images)
            
            print(f"  {category_dir.name}: {len(selected_images)} images")
        
        return images
    
    def _get_test_images(self) -> List[Path]:
        """Get images from test images directory."""
        images = []
        
        if not self.test_images_dir or not self.test_images_dir.exists():
            print(f"Test images directory {self.test_images_dir} does not exist")
            return images
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(self.test_images_dir.glob(ext))
            images.extend(self.test_images_dir.glob(ext.upper()))
        
        return images
    
    def _aggregate_results(self):
        """Aggregate detailed results into summary statistics."""
        if not self.results['detailed_results']:
            return
        
        # Aggregate by model
        model_stats = {}
        for result in self.results['detailed_results']:
            for model_name, model_result in result['models'].items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'total_tests': 0,
                        'confidence_reductions': [],
                        'confidence_reduction_percentages': [],
                        'attack_successes': [],
                        'entropy_changes': [],
                        'psnrs': []
                    }
                
                model_stats[model_name]['total_tests'] += 1
                model_stats[model_name]['confidence_reductions'].append(model_result['confidence_reduction'])
                model_stats[model_name]['confidence_reduction_percentages'].append(model_result['confidence_reduction_percentage'])
                model_stats[model_name]['attack_successes'].append(model_result['attack_success'])
                model_stats[model_name]['entropy_changes'].append(model_result['entropy_increase'])
                model_stats[model_name]['psnrs'].append(model_result['psnr'])
        
        # Calculate averages for each model
        for model_name, stats in model_stats.items():
            self.results['by_model'][model_name] = {
                'total_tests': stats['total_tests'],
                'avg_confidence_reduction': np.mean(stats['confidence_reductions']),
                'avg_confidence_reduction_percentage': np.mean(stats['confidence_reduction_percentages']),
                'confidence_reduction_success_rate': np.mean(stats['confidence_reductions']) > 0,
                'avg_max_confidence_reduction': np.max(stats['confidence_reductions']),
                'avg_entropy_increase': np.mean(stats['entropy_changes']),
                'attack_success_rate': np.mean(stats['attack_successes']),
                'avg_psnr': np.mean(stats['psnrs'])
            }
        
        # Aggregate by method
        method_stats = {}
        for result in self.results['detailed_results']:
            method = result['method']
            if method not in method_stats:
                method_stats[method] = {
                    'total_tests': 0,
                    'confidence_reductions': [],
                    'attack_successes': [],
                    'psnrs': []
                }
            
            method_stats[method]['total_tests'] += 1
            
            # Average across all models for this result
            conf_reductions = [m['confidence_reduction'] for m in result['models'].values()]
            attack_successes = [m['attack_success'] for m in result['models'].values()]
            psnrs = [m['psnr'] for m in result['models'].values()]
            
            method_stats[method]['confidence_reductions'].extend(conf_reductions)
            method_stats[method]['attack_successes'].extend(attack_successes)
            method_stats[method]['psnrs'].extend(psnrs)
        
        for method, stats in method_stats.items():
            self.results['by_method'][method] = {
                'total_tests': stats['total_tests'],
                'avg_confidence_reduction': np.mean(stats['confidence_reductions']),
                'confidence_reduction_success_rate': np.mean(stats['confidence_reductions']) > 0,
                'attack_success_rate': np.mean(stats['attack_successes']),
                'avg_psnr': np.mean(stats['psnrs'])
            }
        
        # Aggregate by strength
        strength_stats = {}
        for result in self.results['detailed_results']:
            strength = result['protection_strength']
            if strength not in strength_stats:
                strength_stats[strength] = {
                    'total_tests': 0,
                    'confidence_reductions': [],
                    'attack_successes': []
                }
            
            strength_stats[strength]['total_tests'] += 1
            
            # Average across all models for this result
            conf_reductions = [m['confidence_reduction'] for m in result['models'].values()]
            attack_successes = [m['attack_success'] for m in result['models'].values()]
            
            strength_stats[strength]['confidence_reductions'].extend(conf_reductions)
            strength_stats[strength]['attack_successes'].extend(attack_successes)
        
        for strength, stats in strength_stats.items():
            self.results['by_strength'][strength] = {
                'total_tests': stats['total_tests'],
                'avg_confidence_reduction': np.mean(stats['confidence_reductions']),
                'confidence_reduction_success_rate': np.mean(stats['confidence_reductions']) > 0,
                'attack_success_rate': np.mean(stats['attack_successes'])
            }
        
        # Aggregate by noise type
        noise_stats = {}
        for result in self.results['detailed_results']:
            noise_type = result['noise_type']
            if noise_type not in noise_stats:
                noise_stats[noise_type] = {
                    'total_tests': 0,
                    'confidence_reductions': [],
                    'attack_successes': [],
                    'psnrs': []
                }
            
            noise_stats[noise_type]['total_tests'] += 1
            
            # Average across all models for this result
            conf_reductions = [m['confidence_reduction'] for m in result['models'].values()]
            attack_successes = [m['attack_success'] for m in result['models'].values()]
            psnrs = [m['psnr'] for m in result['models'].values()]
            
            noise_stats[noise_type]['confidence_reductions'].extend(conf_reductions)
            noise_stats[noise_type]['attack_successes'].extend(attack_successes)
            noise_stats[noise_type]['psnrs'].extend(psnrs)
        
        for noise_type, stats in noise_stats.items():
            self.results['by_noise_type'][noise_type] = {
                'total_tests': stats['total_tests'],
                'avg_confidence_reduction': np.mean(stats['confidence_reductions']),
                'confidence_reduction_success_rate': np.mean(stats['confidence_reductions']) > 0,
                'attack_success_rate': np.mean(stats['attack_successes']),
                'avg_psnr': np.mean(stats['psnrs'])
            }
        
        # Aggregate by category
        category_stats = {}
        for result in self.results['detailed_results']:
            category = result.get('category', 'Unknown')
            if category not in category_stats:
                category_stats[category] = {
                    'total_tests': 0,
                    'confidence_reductions': [],
                    'attack_successes': [],
                    'psnrs': []
                }
            
            category_stats[category]['total_tests'] += 1
            
            # Average across all models for this result
            conf_reductions = [m['confidence_reduction'] for m in result['models'].values()]
            attack_successes = [m['attack_success'] for m in result['models'].values()]
            psnrs = [m['psnr'] for m in result['models'].values()]
            
            category_stats[category]['confidence_reductions'].extend(conf_reductions)
            category_stats[category]['attack_successes'].extend(attack_successes)
            category_stats[category]['psnrs'].extend(psnrs)
        
        for category, stats in category_stats.items():
            self.results['by_category'][category] = {
                'total_tests': stats['total_tests'],
                'avg_confidence_reduction': np.mean(stats['confidence_reductions']),
                'confidence_reduction_success_rate': np.mean(stats['confidence_reductions']) > 0,
                'attack_success_rate': np.mean(stats['attack_successes']),
                'avg_psnr': np.mean(stats['psnrs'])
            }
    
    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"effectiveness_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert results before saving
        serializable_results = convert_numpy_types(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        return output_path


def main():
    """Main function to run the enhanced effectiveness test with command-line arguments."""
    parser = argparse.ArgumentParser(description='MandrAIk Enhanced Effectiveness Test Suite')
    parser.add_argument('--categories', '-c', nargs='+', 
                       help='Specific categories to test (e.g., "Painting" "Drawing")')
    parser.add_argument('--methods', '-m', nargs='+', 
                       choices=['poison', 'fourier'],
                       default=['poison', 'fourier'],
                       help='Protection methods to test (default: poison, fourier)')
    parser.add_argument('--strengths', '-s', nargs='+',
                       choices=['low', 'medium', 'high'],
                       default=['low'],
                       help='Protection strengths to test (default: low)')
    parser.add_argument('--noise-types', '-n', nargs='+',
                       choices=['perlin', 'linear', 'fourier'],
                       default=['perlin'],
                       help='Noise types to test (default: perlin)')
    parser.add_argument('--images-per-category', '-i', type=int, default=30,
                       help='Number of images to test per category (default: 30)')
    parser.add_argument('--test-dataset', '-d', default='test_dataset',
                       help='Path to test dataset directory (default: test_dataset)')
    parser.add_argument('--output-dir', '-o', default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--list-categories', '-l', action='store_true',
                       help='List available categories and exit')
    
    args = parser.parse_args()
    
    print("MandrAIk Enhanced Effectiveness Test Suite")
    print("=" * 50)
    print("Features:")
    print("- Semantic target images")
    print("- mixed7 layer (best performing)")
    print("- Gradient-guided perturbations")
    print("- Art-specific classifiers")
    print("- Comprehensive metrics")
    print("- Dataset support")
    print()
    
    # Initialize tester
    tester = MandrAIkEffectivenessTester(
        test_dataset_dir=args.test_dataset,
        output_dir=args.output_dir
    )
    
    # List categories if requested
    if args.list_categories:
        print("Available categories:")
        for category in tester.art_categories:
            print(f"  - {category}")
        return
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(
        methods=args.methods,
        strengths=args.strengths,
        noise_types=args.noise_types,
        images_per_category=args.images_per_category
    )
    
    # Save results
    tester.save_results()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests: {results['summary']['total_tests']}")
    print(f"Successful protections: {results['summary']['successful_protections']}")
    print(f"Protection success rate: {results['summary']['protection_success_rate']:.2%}")
    
    print("\nBy Model:")
    for model_name, stats in results['by_model'].items():
        print(f"  {model_name}:")
        print(f"    Attack success rate: {stats['attack_success_rate']:.2%}")
        print(f"    Avg confidence reduction: {stats['avg_confidence_reduction']:.4f}")
        print(f"    Avg PSNR: {stats['avg_psnr']:.2f}")
    
    print("\nBy Method:")
    for method_name, stats in results['by_method'].items():
        print(f"  {method_name}:")
        print(f"    Attack success rate: {stats['attack_success_rate']:.2%}")
        print(f"    Avg confidence reduction: {stats['avg_confidence_reduction']:.4f}")
        print(f"    Avg PSNR: {stats['avg_psnr']:.2f}")


if __name__ == "__main__":
    main() 