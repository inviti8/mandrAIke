#!/usr/bin/env python3
"""
MandrAIk Effectiveness Test Suite

This test evaluates the effectiveness of MandrAIk in creating adversarial images
that prevent AI image recognition systems from correctly classifying protected images.

Uses modern standards:
- Multiple state-of-the-art image classification models
- Standard adversarial attack evaluation metrics
- Real-world image categories
- Comprehensive statistical analysis
- Target-based perturbation using images from target_images folder
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import random

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0, vit_b_16
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights

# Import MandrAIk
from MandrAIk import MandrAIk

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class ModernImageClassifier:
    """Wrapper for modern image classification models."""
    
    def __init__(self, model_name: str = 'inception_v3'):
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.class_names = None
        self.framework = None
        self.input_size = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified model."""
        try:
            if self.model_name == 'inception_v3':
                self.model = tf.keras.applications.InceptionV3(
                    weights='imagenet',
                    include_top=True
                )
                self.preprocess = tf.keras.applications.inception_v3.preprocess_input
                self.class_names = self._get_imagenet_class_names()
                self.framework = 'tensorflow'
                self.input_size = (299, 299)
                
            elif self.model_name == 'resnet50':
                self.model = tf.keras.applications.ResNet50(
                    weights='imagenet',
                    include_top=True
                )
                self.preprocess = tf.keras.applications.resnet50.preprocess_input
                self.class_names = self._get_imagenet_class_names()
                self.framework = 'tensorflow'
                self.input_size = (224, 224)
                
            elif self.model_name == 'efficientnet_b0':
                self.model = tf.keras.applications.EfficientNetB0(
                    weights='imagenet',
                    include_top=True
                )
                self.preprocess = tf.keras.applications.efficientnet.preprocess_input
                self.class_names = self._get_imagenet_class_names()
                self.framework = 'tensorflow'
                self.input_size = (224, 224)
                
            elif self.model_name == 'pytorch_resnet50':
                self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.model.eval()
                self.preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.class_names = self._get_imagenet_class_names()
                self.framework = 'pytorch'
                self.input_size = (224, 224)
                
            elif self.model_name == 'pytorch_efficientnet':
                self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                self.model.eval()
                self.preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.class_names = self._get_imagenet_class_names()
                self.framework = 'pytorch'
                self.input_size = (224, 224)
                
            elif self.model_name == 'pytorch_vit':
                try:
                    self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                    self.model.eval()
                    self.preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    self.class_names = self._get_imagenet_class_names()
                    self.framework = 'pytorch'
                    self.input_size = (224, 224)
                except Exception as e:
                    print(f"Warning: Could not load Vision Transformer model: {e}")
                    print("Falling back to ResNet50...")
                    # Fallback to ResNet50
                    self.model_name = 'pytorch_resnet50'
                    self._load_model()
                    return
                    
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")
            # Fallback to InceptionV3 if available
            if self.model_name != 'inception_v3':
                print("Falling back to InceptionV3...")
                self.model_name = 'inception_v3'
                self._load_model()
    
    def _get_imagenet_class_names(self) -> List[str]:
        """Get ImageNet class names."""
        try:
            # Try to load from keras
            return tf.keras.applications.imagenet_utils.decode_predictions(
                np.zeros((1, 1000)), top=1
            )[0][0][1]
        except:
            # Fallback to basic class names
            return [f"class_{i}" for i in range(1000)]
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k predictions for an image."""
        try:
            if self.framework == 'tensorflow':
                return self._predict_tensorflow(image_path, top_k)
            else:
                return self._predict_pytorch(image_path, top_k)
        except Exception as e:
            print(f"Prediction error for {image_path}: {e}")
            return [("error", 0.0)]
    
    def _predict_tensorflow(self, image_path: str, top_k: int) -> List[Tuple[str, float]]:
        """TensorFlow prediction."""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.input_size)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess(x)
        
        preds = self.model.predict(x, verbose=0)
        decoded = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=top_k)[0]
        
        return [(class_name, float(confidence)) for _, class_name, confidence in decoded]
    
    def _predict_pytorch(self, image_path: str, top_k: int) -> List[Tuple[str, float]]:
        """PyTorch prediction."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
            results.append((class_name, confidence))
        
        return results

class MandrAIkEffectivenessTester:
    """Comprehensive effectiveness tester for MandrAIk."""
    
    def __init__(self, test_images_dir: str = "test_images", output_dir: str = "test_results", target_images_dir: str = "target_images"):
        self.test_images_dir = Path(test_images_dir)
        self.output_dir = Path(output_dir)
        self.target_images_dir = Path(target_images_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load target images for perturbation
        self.target_images = self._load_target_images()
        
        # Initialize models
        self.models = {
            'inception_v3': ModernImageClassifier('inception_v3'),
            'resnet50': ModernImageClassifier('resnet50'),
            'efficientnet_b0': ModernImageClassifier('efficientnet_b0'),
            'pytorch_resnet50': ModernImageClassifier('pytorch_resnet50'),
            'pytorch_efficientnet': ModernImageClassifier('pytorch_efficientnet'),
            'pytorch_vit': ModernImageClassifier('pytorch_vit')
        }
        
        # Remove failed models
        self.models = {k: v for k, v in self.models.items() if v.model is not None}
        
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        print(f"Loaded {len(self.target_images)} target images: {[p.name for p in self.target_images]}")
    
    def _load_target_images(self) -> List[Path]:
        """Load target images from target_images directory."""
        target_images = []
        if self.target_images_dir.exists():
            # Load both target_image.jpg and target_image2.jpg for comprehensive testing
            target_files = ["target_image.jpg", "target_image2.jpg"]
            for target_file in target_files:
                target_path = self.target_images_dir / target_file
                if target_path.exists():
                    target_images.append(target_path)
                    print(f"Loaded target image: {target_file}")
                else:
                    print(f"Warning: Target image {target_file} not found")
        
        if not target_images:
            print("Warning: No target images found in target_images directory")
        
        return target_images
    
    def load_test_images(self) -> List[Path]:
        """Load test images from the test directory."""
        test_images = []
        
        if self.test_images_dir.exists():
            # Load existing test images
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(self.test_images_dir.glob(ext))
        
        if not test_images:
            print("No test images found. Creating sample images...")
            test_images = self._create_sample_images()
        
        return test_images
    
    def _create_sample_images(self) -> List[Path]:
        """Create sample test images if none exist."""
        self.test_images_dir.mkdir(exist_ok=True)
        
        # Create sample images with different characteristics
        sample_images = []
        
        # Create a simple geometric pattern
        img1 = np.zeros((224, 224, 3), dtype=np.uint8)
        img1[50:150, 50:150] = [255, 0, 0]  # Red square
        img1_path = self.test_images_dir / "sample_red_square.jpg"
        Image.fromarray(img1).save(img1_path)
        sample_images.append(img1_path)
        
        # Create a gradient
        img2 = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            intensity = int(255 * i / 224)
            img2[i, :] = [intensity, intensity, intensity]
        img2_path = self.test_images_dir / "sample_gradient.jpg"
        Image.fromarray(img2).save(img2_path)
        sample_images.append(img2_path)
        
        # Create a striped pattern
        img3 = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, 20):
            img3[i:i+10, :] = [0, 255, 0]  # Green stripes
        img3_path = self.test_images_dir / "sample_stripes.jpg"
        Image.fromarray(img3).save(img3_path)
        sample_images.append(img3_path)
        
        print(f"Created {len(sample_images)} sample test images")
        return sample_images
    
    def calculate_image_quality(self, original_path: Path, protected_path: Path) -> Dict[str, float]:
        """Calculate image quality metrics."""
        try:
            original = cv2.imread(str(original_path))
            protected = cv2.imread(str(protected_path))
            
            if original is None or protected is None:
                return {"psnr": 0.0, "ssim": 0.0, "mse": float('inf')}
            
            # Ensure same size
            if original.shape != protected.shape:
                protected = cv2.resize(protected, (original.shape[1], original.shape[0]))
            
            # Convert to grayscale for SSIM
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            protected_gray = cv2.cvtColor(protected, cv2.COLOR_BGR2GRAY)
            
            # Calculate MSE
            mse = np.mean((original.astype(np.float64) - protected.astype(np.float64)) ** 2)
            
            # Calculate PSNR
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # Calculate SSIM (simplified)
            mu_x = np.mean(original_gray)
            mu_y = np.mean(protected_gray)
            sigma_x = np.std(original_gray)
            sigma_y = np.std(protected_gray)
            sigma_xy = np.mean((original_gray - mu_x) * (protected_gray - mu_y))
            
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
            
            return {
                "psnr": float(psnr),
                "ssim": float(ssim),
                "mse": float(mse)
            }
        except Exception as e:
            print(f"Error calculating image quality: {e}")
            return {"psnr": 0.0, "ssim": 0.0, "mse": float('inf')}
    
    def test_single_image(self, image_path: Path, protection_strength: str = 'medium', target_image: Path = None, method: str = 'poison') -> Dict:
        """Test protection on a single image with focus on confidence reduction."""
        # Select target image if not provided
        if target_image is None:
            target_image = random.choice(self.target_images) if self.target_images else None
        
        if target_image is None:
            print(f"No target images available for {image_path}")
            return {}
        
        # For hallucinogen method, we need two target images
        if method == 'hallucinogen':
            if len(self.target_images) < 2:
                print(f"Hallucinogen method requires at least 2 target images, but only {len(self.target_images)} available")
                return {}
            target_image2 = random.choice([t for t in self.target_images if t != target_image])
        
        # Map protection strength to numerical value
        strength_map = {
            'low': 0.1,
            'medium': 0.15,
            'high': 0.25
        }
        strength_value = strength_map.get(protection_strength, 0.15)
        
        # Create output path
        if method == 'hallucinogen':
            output_filename = f"protected_{image_path.stem}_{target_image.stem}_{target_image2.stem}_{method}_{protection_strength}.jpg"
        else:
            output_filename = f"protected_{image_path.stem}_{target_image.stem}_{method}_{protection_strength}.jpg"
        output_path = self.output_dir / output_filename
        
        # Initialize MandrAIk
        mandrAIk = MandrAIk(
            model_name='InceptionV3',
            steps=3,
            step_size=0.5,
            num_ocataves=3,
            octave_scale=2.5,
            layer_name='mixed1'
        )
        
        # Apply protection
        try:
            start_time = time.time()
            
            if method == 'poison':
                mandrAIk.poison(
                    image_path=str(image_path),
                    target_image_path=str(target_image),
                    output_path=str(output_path),
                    protection_strength=strength_value
                )
            elif method == 'hallucinogen':
                mandrAIk.hallucinogen(
                    image_path=str(image_path),
                    target_image_path1=str(target_image),
                    target_image_path2=str(target_image2),
                    output_path=str(output_path),
                    protection_strength=strength_value
                )
            else:
                raise ValueError(f"Unknown protection method: {method}")
            
            protection_time = time.time() - start_time
            
            # Test with all models
            results = {
                'image_path': str(image_path),
                'target_image': str(target_image),
                'method': method,
                'protected_path': str(output_path),
                'protection_strength': protection_strength,
                'strength_value': strength_value,
                'protection_time': protection_time,
                'models': {}
            }
            
            # Add second target for hallucinogen
            if method == 'hallucinogen':
                results['target_image2'] = str(target_image2)
            
            # Get original predictions
            original_predictions = {}
            for model_name, model in self.models.items():
                try:
                    original_predictions[model_name] = model.predict(str(image_path))
                except Exception as e:
                    print(f"Error getting original predictions for {model_name}: {e}")
                    original_predictions[model_name] = [("error", 0.0)]
            
            # Get protected predictions
            for model_name, model in self.models.items():
                try:
                    protected_predictions = model.predict(str(output_path))
                    
                    # Calculate confidence reduction metrics
                    original_top1 = original_predictions[model_name][0]
                    protected_top1 = protected_predictions[0]
                    
                    # Primary metric: Confidence reduction effectiveness
                    confidence_reduction = original_top1[1] - protected_top1[1]
                    confidence_reduction_percentage = (confidence_reduction / original_top1[1]) * 100
                    
                    # Secondary metrics
                    attack_success = original_top1[0] != protected_top1[0]
                    top5_success = original_top1[0] not in [p[0] for p in protected_predictions]
                    
                    # Calculate confidence distribution metrics
                    original_confidence_entropy = self._calculate_confidence_entropy(original_predictions[model_name])
                    protected_confidence_entropy = self._calculate_confidence_entropy(protected_predictions)
                    
                    # Check if confidence was actually reduced (primary goal)
                    confidence_reduced = confidence_reduction > 0
                    
                    # Calculate max confidence reduction (how much the top prediction dropped)
                    max_confidence_reduction = original_top1[1] - max([p[1] for p in protected_predictions])
                    
                    results['models'][model_name] = {
                        'original_top1': original_top1,
                        'protected_top1': protected_top1,
                        'original_top5': original_predictions[model_name],
                        'protected_top5': protected_predictions,
                        
                        # Primary confidence reduction metrics
                        'confidence_reduction': confidence_reduction,
                        'confidence_reduction_percentage': confidence_reduction_percentage,
                        'confidence_reduced': confidence_reduced,
                        'max_confidence_reduction': max_confidence_reduction,
                        
                        # Secondary attack metrics
                        'attack_success': attack_success,
                        'top5_success': top5_success,
                        
                        # Confidence distribution metrics
                        'original_confidence_entropy': original_confidence_entropy,
                        'protected_confidence_entropy': protected_confidence_entropy,
                        'entropy_increase': protected_confidence_entropy - original_confidence_entropy
                    }
                    
                except Exception as e:
                    print(f"Error testing {model_name} on protected image: {e}")
                    results['models'][model_name] = {
                        'error': str(e)
                    }
            
            # Calculate image quality
            quality_metrics = self.calculate_image_quality(image_path, output_path)
            results['quality_metrics'] = quality_metrics
            
            return results
            
        except Exception as e:
            print(f"Error protecting image {image_path}: {e}")
            return {'error': str(e), 'image_path': str(image_path)}
    
    def _calculate_confidence_entropy(self, predictions: List[Tuple[str, float]]) -> float:
        """Calculate entropy of confidence distribution."""
        confidences = [p[1] for p in predictions]
        # Normalize to probability distribution
        total = sum(confidences)
        if total == 0:
            return 0.0
        probabilities = [c / total for c in confidences]
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def run_comprehensive_test(self, protection_strengths: List[str] = None, methods: List[str] = None) -> Dict:
        """Run comprehensive effectiveness test."""
        if protection_strengths is None:
            protection_strengths = ['low', 'medium', 'high']
        
        if methods is None:
            methods = ['poison', 'hallucinogen']
        
        print(f"Starting comprehensive test with strengths: {protection_strengths}")
        print(f"Testing methods: {methods}")
        print(f"Using {len(self.target_images)} target images")
        
        # Load test images
        test_images = self.load_test_images()
        print(f"Testing with {len(test_images)} images")
        
        all_results = []
        
        # Test each image with each strength, method, and target
        for image_path in test_images:
            print(f"\nTesting image: {image_path.name}")
            
            for method in methods:
                print(f"  Method: {method}")
                
                for strength in protection_strengths:
                    for target_image in self.target_images:
                        print(f"    Strength: {strength}, Target: {target_image.name}")
                        
                        result = self.test_single_image(
                            image_path=image_path,
                            protection_strength=strength,
                            target_image=target_image,
                            method=method
                        )
                        
                        if result and 'error' not in result:
                            all_results.append(result)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Save results
        self._save_results(aggregated_results)
        
        # Generate report
        self._generate_report(aggregated_results)
        
        return aggregated_results
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate test results with focus on confidence reduction effectiveness."""
        if not results:
            return {}
        
        aggregated = {
            'summary': {},
            'by_model': {},
            'by_method': {},
            'by_strength': {},
            'by_target': {},
            'detailed_results': results
        }
        
        # Overall statistics
        total_tests = len(results)
        successful_protections = sum(1 for r in results if 'error' not in r)
        
        aggregated['summary'] = {
            'total_tests': total_tests,
            'successful_protections': successful_protections,
            'protection_success_rate': successful_protections / total_tests if total_tests > 0 else 0
        }
        
        # Aggregate by model - focus on confidence reduction
        for model_name in self.models.keys():
            model_results = [r for r in results if 'error' not in r and model_name in r.get('models', {})]
            
            if model_results:
                # Confidence reduction metrics (primary focus)
                confidence_reductions = [r['models'][model_name].get('confidence_reduction', 0) for r in model_results]
                confidence_reduction_percentages = [r['models'][model_name].get('confidence_reduction_percentage', 0) for r in model_results]
                confidence_reduced_count = sum(1 for r in model_results if r['models'][model_name].get('confidence_reduced', False))
                max_confidence_reductions = [r['models'][model_name].get('max_confidence_reduction', 0) for r in model_results]
                
                # Entropy metrics
                entropy_increases = [r['models'][model_name].get('entropy_increase', 0) for r in model_results]
                
                # Secondary attack metrics
                attack_successes = sum(1 for r in model_results if r['models'][model_name].get('attack_success', False))
                
                # Quality metrics
                avg_psnr = np.mean([r.get('quality_metrics', {}).get('psnr', 0) for r in model_results])
                
                aggregated['by_model'][model_name] = {
                    'total_tests': len(model_results),
                    
                    # Primary: Confidence reduction effectiveness
                    'avg_confidence_reduction': np.mean(confidence_reductions),
                    'avg_confidence_reduction_percentage': np.mean(confidence_reduction_percentages),
                    'confidence_reduction_success_rate': confidence_reduced_count / len(model_results),
                    'avg_max_confidence_reduction': np.mean(max_confidence_reductions),
                    
                    # Confidence distribution
                    'avg_entropy_increase': np.mean(entropy_increases),
                    
                    # Secondary: Attack success
                    'attack_success_rate': attack_successes / len(model_results),
                    
                    # Quality
                    'avg_psnr': avg_psnr
                }
        
        # Aggregate by method - focus on confidence reduction
        for method in ['poison', 'hallucinogen']:
            method_results = [r for r in results if r.get('method') == method]
            
            if method_results:
                # Confidence reduction metrics
                confidence_reductions = [r['models'][m].get('confidence_reduction', 0) 
                                       for r in method_results 
                                       for m in r['models'] if 'error' not in r['models'][m]]
                confidence_reduced_count = sum(1 for r in method_results 
                                             for m in r['models'] if 'error' not in r['models'][m] 
                                             and r['models'][m].get('confidence_reduced', False))
                
                # Attack success (secondary)
                attack_successes = sum(1 for r in method_results 
                                     if any(r['models'][m].get('attack_success', False) 
                                           for m in r['models'] if 'error' not in r['models'][m]))
                
                # Quality metrics
                avg_psnr = np.mean([r.get('quality_metrics', {}).get('psnr', 0) for r in method_results])
                
                aggregated['by_method'][method] = {
                    'total_tests': len(method_results),
                    'avg_confidence_reduction': np.mean(confidence_reductions),
                    'confidence_reduction_success_rate': confidence_reduced_count / len(confidence_reductions) if confidence_reductions else 0,
                    'attack_success_rate': attack_successes / len(method_results),
                    'avg_psnr': avg_psnr
                }
        
        # Aggregate by strength - focus on confidence reduction
        for strength in ['low', 'medium', 'high']:
            strength_results = [r for r in results if r.get('protection_strength') == strength]
            
            if strength_results:
                # Confidence reduction metrics
                confidence_reductions = [r['models'][m].get('confidence_reduction', 0) 
                                       for r in strength_results 
                                       for m in r['models'] if 'error' not in r['models'][m]]
                confidence_reduced_count = sum(1 for r in strength_results 
                                             for m in r['models'] if 'error' not in r['models'][m] 
                                             and r['models'][m].get('confidence_reduced', False))
                
                # Attack success (secondary)
                attack_successes = sum(1 for r in strength_results 
                                     if any(r['models'][m].get('attack_success', False) 
                                           for m in r['models'] if 'error' not in r['models'][m]))
                
                aggregated['by_strength'][strength] = {
                    'total_tests': len(strength_results),
                    'avg_confidence_reduction': np.mean(confidence_reductions),
                    'confidence_reduction_success_rate': confidence_reduced_count / len(confidence_reductions) if confidence_reductions else 0,
                    'attack_success_rate': attack_successes / len(strength_results)
                }
        
        # Aggregate by target - focus on confidence reduction
        for target_path in self.target_images:
            target_results = [r for r in results if r.get('target_image') == str(target_path)]
            
            if target_results:
                # Confidence reduction metrics
                confidence_reductions = [r['models'][m].get('confidence_reduction', 0) 
                                       for r in target_results 
                                       for m in r['models'] if 'error' not in r['models'][m]]
                confidence_reduced_count = sum(1 for r in target_results 
                                             for m in r['models'] if 'error' not in r['models'][m] 
                                             and r['models'][m].get('confidence_reduced', False))
                
                # Attack success (secondary)
                attack_successes = sum(1 for r in target_results 
                                     if any(r['models'][m].get('attack_success', False) 
                                           for m in r['models'] if 'error' not in r['models'][m]))
                
                aggregated['by_target'][target_path.name] = {
                    'total_tests': len(target_results),
                    'avg_confidence_reduction': np.mean(confidence_reductions),
                    'confidence_reduction_success_rate': confidence_reduced_count / len(confidence_reductions) if confidence_reductions else 0,
                    'attack_success_rate': attack_successes / len(target_results)
                }
        
        return aggregated
    
    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert numpy types to native Python types
        results_serializable = json.loads(json.dumps(results, default=convert_numpy))
        
        output_file = self.output_dir / "effectiveness_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to: {output_file}")
    
    def _generate_report(self, results: Dict):
        """Generate a comprehensive report focused on confidence reduction effectiveness."""
        report_file = self.output_dir / "effectiveness_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("MandrAIk Confidence Reduction Effectiveness Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            summary = results.get('summary', {})
            f.write(f"Total Tests: {summary.get('total_tests', 0)}\n")
            f.write(f"Successful Protections: {summary.get('successful_protections', 0)}\n")
            f.write(f"Protection Success Rate: {summary.get('protection_success_rate', 0):.2%}\n\n")
            
            # By Model - Primary focus on confidence reduction
            f.write("Results by Model (Confidence Reduction Focus):\n")
            f.write("-" * 50 + "\n")
            for model_name, model_results in results.get('by_model', {}).items():
                f.write(f"{model_name}:\n")
                f.write(f"  Confidence Reduction Success Rate: {model_results['confidence_reduction_success_rate']:.2%}\n")
                f.write(f"  Avg Confidence Reduction: {model_results['avg_confidence_reduction']:.3f}\n")
                f.write(f"  Avg Confidence Reduction %: {model_results['avg_confidence_reduction_percentage']:.1f}%\n")
                f.write(f"  Avg Max Confidence Reduction: {model_results['avg_max_confidence_reduction']:.3f}\n")
                f.write(f"  Avg Entropy Increase: {model_results['avg_entropy_increase']:.3f}\n")
                f.write(f"  Attack Success Rate: {model_results['attack_success_rate']:.2%}\n")
                f.write(f"  Avg PSNR: {model_results['avg_psnr']:.2f}\n\n")
            
            # By Method - Primary focus on confidence reduction
            f.write("Results by Protection Method (Confidence Reduction Focus):\n")
            f.write("-" * 55 + "\n")
            for method, method_results in results.get('by_method', {}).items():
                f.write(f"{method}:\n")
                f.write(f"  Confidence Reduction Success Rate: {method_results['confidence_reduction_success_rate']:.2%}\n")
                f.write(f"  Avg Confidence Reduction: {method_results['avg_confidence_reduction']:.3f}\n")
                f.write(f"  Attack Success Rate: {method_results['attack_success_rate']:.2%}\n")
                f.write(f"  Avg PSNR: {method_results['avg_psnr']:.2f}\n\n")
            
            # By Strength - Primary focus on confidence reduction
            f.write("Results by Protection Strength (Confidence Reduction Focus):\n")
            f.write("-" * 55 + "\n")
            for strength, strength_results in results.get('by_strength', {}).items():
                f.write(f"{strength}:\n")
                f.write(f"  Confidence Reduction Success Rate: {strength_results['confidence_reduction_success_rate']:.2%}\n")
                f.write(f"  Avg Confidence Reduction: {strength_results['avg_confidence_reduction']:.3f}\n")
                f.write(f"  Attack Success Rate: {strength_results['attack_success_rate']:.2%}\n\n")
            
            # By Target - Primary focus on confidence reduction
            f.write("Results by Target Image (Confidence Reduction Focus):\n")
            f.write("-" * 45 + "\n")
            for target_name, target_results in results.get('by_target', {}).items():
                f.write(f"{target_name}:\n")
                f.write(f"  Confidence Reduction Success Rate: {target_results['confidence_reduction_success_rate']:.2%}\n")
                f.write(f"  Avg Confidence Reduction: {target_results['avg_confidence_reduction']:.3f}\n")
                f.write(f"  Attack Success Rate: {target_results['attack_success_rate']:.2%}\n\n")
            
            # Key Insights
            f.write("Key Insights:\n")
            f.write("-" * 15 + "\n")
            
            # Find best performing models for confidence reduction
            model_results = results.get('by_model', {})
            if model_results:
                best_confidence_reduction = max(model_results.items(), 
                                              key=lambda x: x[1]['confidence_reduction_success_rate'])
                best_avg_reduction = max(model_results.items(), 
                                       key=lambda x: x[1]['avg_confidence_reduction'])
                
                f.write(f"Best Confidence Reduction Success: {best_confidence_reduction[0]} ({best_confidence_reduction[1]['confidence_reduction_success_rate']:.2%})\n")
                f.write(f"Best Avg Confidence Reduction: {best_avg_reduction[0]} ({best_avg_reduction[1]['avg_confidence_reduction']:.3f})\n")
            
            # Find best performing targets
            target_results = results.get('by_target', {})
            if target_results:
                best_target = max(target_results.items(), 
                                key=lambda x: x[1]['confidence_reduction_success_rate'])
                f.write(f"Best Target for Confidence Reduction: {best_target[0]} ({best_target[1]['confidence_reduction_success_rate']:.2%})\n")
            
            # Find best strength
            strength_results = results.get('by_strength', {})
            if strength_results:
                best_strength = max(strength_results.items(), 
                                  key=lambda x: x[1]['confidence_reduction_success_rate'])
                f.write(f"Best Protection Strength: {best_strength[0]} ({best_strength[1]['confidence_reduction_success_rate']:.2%})\n")
        
        print(f"Report saved to: {report_file}")
    
    def _create_visualizations(self, results: Dict):
        """Create visualizations of the results."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('MandrAIk Effectiveness Test Results', fontsize=16)
            
            # 1. Attack success rate by model
            if results.get('by_model'):
                models = list(results['by_model'].keys())
                success_rates = [results['by_model'][m]['attack_success_rate'] for m in models]
                
                axes[0, 0].bar(models, success_rates, color='skyblue')
                axes[0, 0].set_title('Attack Success Rate by Model')
                axes[0, 0].set_ylabel('Success Rate')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].set_ylim(0, 1)
                
                # Add percentage labels
                for i, v in enumerate(success_rates):
                    axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center')
            
            # 2. Confidence drop by model
            if results.get('by_model'):
                confidence_drops = [results['by_model'][m]['avg_confidence_reduction'] for m in models]
                
                axes[0, 1].bar(models, confidence_drops, color='lightcoral')
                axes[0, 1].set_title('Average Confidence Drop by Model')
                axes[0, 1].set_ylabel('Confidence Drop')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for i, v in enumerate(confidence_drops):
                    axes[0, 1].text(i, v + 0.01 if v > 0 else v - 0.01, f'{v:.3f}', ha='center')
            
            # 3. Attack success rate by protection strength
            if results.get('by_strength'):
                strengths = list(results['by_strength'].keys())
                strength_success_rates = [results['by_strength'][s]['attack_success_rate'] for s in strengths]
                
                axes[1, 0].bar(strengths, strength_success_rates, color='lightgreen')
                axes[1, 0].set_title('Attack Success Rate by Protection Strength')
                axes[1, 0].set_ylabel('Success Rate')
                axes[1, 0].set_ylim(0, 1)
                
                # Add percentage labels
                for i, v in enumerate(strength_success_rates):
                    axes[1, 0].text(i, v + 0.01, f'{v:.1%}', ha='center')
            
            # 4. Image quality (PSNR) by model
            if results.get('by_model'):
                psnr_values = [results['by_model'][m]['avg_psnr'] for m in models]
                
                axes[1, 1].bar(models, psnr_values, color='gold')
                axes[1, 1].set_title('Average PSNR by Model')
                axes[1, 1].set_ylabel('PSNR (dB)')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for i, v in enumerate(psnr_values):
                    axes[1, 1].text(i, v + 0.5, f'{v:.1f}', ha='center')
            
            plt.tight_layout()
            
            # Save the plot
            plot_file = self.output_dir / "effectiveness_visualization.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to: {plot_file}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")

def main():
    """Main function to run the effectiveness test."""
    print("MandrAIk Effectiveness Test Suite")
    print("=" * 40)
    
    # Initialize tester
    tester = MandrAIkEffectivenessTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Create visualizations
    tester._create_visualizations(results)
    
    print("\nTest completed!")
    print(f"Results saved to: {tester.output_dir}")

if __name__ == "__main__":
    main() 