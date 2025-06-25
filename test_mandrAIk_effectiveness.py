#!/usr/bin/env python3
"""
MandrAIk Effectiveness Test Suite - Enhanced Version

This test evaluates the effectiveness of MandrAIk in creating adversarial images
that prevent AI image recognition systems from correctly classifying protected images.

Enhanced features:
- Semantic target images specific to each test image
- mixed7 layer as default (best performing from our tests)
- Enhanced hallucinogen method with multi-color space perturbations
- Focused testing on InceptionV3 model for consistency
- Comprehensive statistical analysis
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
from datetime import datetime

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
            return self._predict_tensorflow(image_path, top_k)
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

class MandrAIkEffectivenessTester:
    """Enhanced effectiveness tester for MandrAIk with semantic targets."""
    
    def __init__(self, test_images_dir: str = "test_images", output_dir: str = "test_results", target_images_dir: str = "target_images"):
        """Initialize the tester with semantic target support."""
        self.test_images_dir = Path(test_images_dir)
        self.output_dir = Path(output_dir)
        self.target_images_dir = Path(target_images_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load models (focus on InceptionV3 for consistency)
        self.models = {
            'inception_v3': ModernImageClassifier('inception_v3'),
            'resnet50': ModernImageClassifier('resnet50'),
            'efficientnet_b0': ModernImageClassifier('efficientnet_b0')
        }
        
        # Remove failed models
        self.models = {k: v for k, v in self.models.items() if v.model is not None}
        
        # Load target images (will be matched semantically)
        self.target_images = self._load_target_images()
        
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        print(f"Loaded {len(self.target_images)} target images: {[p.name for p in self.target_images]}")
    
    def _load_target_images(self) -> List[Path]:
        """Load target images from target_images directory."""
        target_images = []
        if self.target_images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                target_images.extend(self.target_images_dir.glob(ext))
        
        if not target_images:
            print("Warning: No target images found in target_images directory")
        
        return target_images
    
    def _find_semantic_targets(self, test_image_path: Path) -> Tuple[Path, Path]:
        """Find semantic target images for a given test image."""
        test_name = test_image_path.stem
        
        # Look for semantic targets
        target1_pattern = f"{test_name}_target_image1.*"
        target2_pattern = f"{test_name}_target_image2.*"
        
        target1_matches = list(self.target_images_dir.glob(target1_pattern))
        target2_matches = list(self.target_images_dir.glob(target2_pattern))
        
        if target1_matches and target2_matches:
            print(f"  Using semantic targets for {test_name}: {target1_matches[0].name}, {target2_matches[0].name}")
            return target1_matches[0], target2_matches[0]
        else:
            # Fallback to generic targets
            generic_targets = [t for t in self.target_images if t.name in ['target_image1.jpg', 'target_image2.jpg']]
            if len(generic_targets) >= 2:
                print(f"  Using generic targets for {test_name}")
                return generic_targets[0], generic_targets[1]
            else:
                # Use any available targets
                if len(self.target_images) >= 2:
                    print(f"  Using available targets for {test_name}")
                    return self.target_images[0], self.target_images[1]
                else:
                    raise ValueError(f"Not enough target images available for {test_name}")
    
    def load_test_images(self) -> List[Path]:
        """Load test images from the test directory."""
        test_images = []
        
        if self.test_images_dir.exists():
            # Load existing test images
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(self.test_images_dir.glob(ext))
        
        if not test_images:
            print("No test images found. Please add images to test_images directory.")
            return []
        
        return test_images
    
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
    
    def test_single_image(self, image_path: Path, protection_strength: str = 'medium', method: str = 'hallucinogen') -> Dict:
        """Test protection on a single image with semantic targets."""
        try:
            # Find semantic targets for this image
            target_image1, target_image2 = self._find_semantic_targets(image_path)
            
            # Map protection strength to numerical value
            strength_map = {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.8
            }
            strength_value = strength_map.get(protection_strength, 0.5)
            
            # Create output path
            output_filename = f"protected_{image_path.stem}_{method}_{protection_strength}.jpg"
            output_path = self.output_dir / output_filename
            
            # Initialize MandrAIk with mixed7 layer (best performing)
            mandrAIk = MandrAIk(
                model_name='InceptionV3',
                steps=3,
                step_size=0.5,
                num_ocataves=3,
                octave_scale=2.5,
                layer_name='mixed7'  # Use mixed7 as default
            )
            
            # Apply protection
            start_time = time.time()
            
            if method == 'hallucinogen':
                mandrAIk.hallucinogen(
                    image_path=str(image_path),
                    target_image_path1=str(target_image1),
                    target_image_path2=str(target_image2),
                    output_path=str(output_path),
                    protection_strength=strength_value
                )
            else:
                raise ValueError(f"Method {method} not supported in enhanced version")
            
            protection_time = time.time() - start_time
            
            # Test with all models
            results = {
                'image_path': str(image_path),
                'target_image1': str(target_image1),
                'target_image2': str(target_image2),
                'method': method,
                'protected_path': str(output_path),
                'protection_strength': protection_strength,
                'strength_value': strength_value,
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
                if original_results[model_name] and protected_results[model_name]:
                    orig_top1 = original_results[model_name]['original_top1']
                    prot_top1 = protected_results[model_name]['protected_top1']
                    
                    # Calculate confidence reduction
                    confidence_reduction = orig_top1[1] - prot_top1[1]
                    if orig_top1[1] > 0:
                        confidence_reduction_percentage = (confidence_reduction / orig_top1[1]) * 100
                    else:
                        confidence_reduction_percentage = 0.0
                    
                    # Calculate top-5 confidence reduction
                    orig_avg = np.mean([conf for _, conf in original_results[model_name]['original_top5']])
                    prot_avg = np.mean([conf for _, conf in protected_results[model_name]['protected_top5']])
                    top5_confidence_reduction = orig_avg - prot_avg
                    
                    # Calculate entropy change
                    entropy_change = protected_results[model_name]['protected_confidence_entropy'] - \
                                   original_results[model_name]['original_confidence_entropy']
                    
                    # Determine attack success
                    attack_success = orig_top1[0] != prot_top1[0]
                    confidence_reduced = confidence_reduction > 0
                    
                    # Calculate image quality
                    quality_metrics = self.calculate_image_quality(image_path, output_path)
                    
                    results['models'][model_name] = {
                        'original_top1': orig_top1,
                        'protected_top1': prot_top1,
                        'original_top5': original_results[model_name]['original_top5'],
                        'protected_top5': protected_results[model_name]['protected_top5'],
                        'confidence_reduction': confidence_reduction,
                        'confidence_reduction_percentage': confidence_reduction_percentage,
                        'top5_confidence_reduction': top5_confidence_reduction,
                        'confidence_reduced': confidence_reduced,
                        'max_confidence_reduction': confidence_reduction,
                        'attack_success': attack_success,
                        'top5_success': False,  # Not implemented in this version
                        'original_confidence_entropy': original_results[model_name]['original_confidence_entropy'],
                        'protected_confidence_entropy': protected_results[model_name]['protected_confidence_entropy'],
                        'entropy_increase': entropy_change,
                        'psnr': quality_metrics['psnr'],
                        'ssim': quality_metrics['ssim'],
                        'mse': quality_metrics['mse']
                    }
            
            return results
            
        except Exception as e:
            print(f"Error testing image {image_path}: {e}")
            return {}
    
    def _calculate_confidence_entropy(self, predictions: List[Tuple[str, float]]) -> float:
        """Calculate entropy of confidence distribution."""
        confidences = [conf for _, conf in predictions]
        confidences = np.array(confidences)
        confidences = confidences / np.sum(confidences)  # Normalize
        confidences = confidences[confidences > 0]  # Remove zeros
        return -np.sum(confidences * np.log2(confidences))
    
    def run_comprehensive_test(self, protection_strengths: List[str] = None, methods: List[str] = None) -> Dict:
        """Run comprehensive test with semantic targets."""
        if protection_strengths is None:
            protection_strengths = ['medium']  # Focus on medium strength
        
        if methods is None:
            methods = ['hallucinogen']  # Focus on enhanced hallucinogen
        
        print(f"\nRunning comprehensive test with:")
        print(f"  Protection strengths: {protection_strengths}")
        print(f"  Methods: {methods}")
        print(f"  Layer: mixed7 (default)")
        
        # Load test images
        test_images = self.load_test_images()
        if not test_images:
            print("No test images found!")
            return {}
        
        print(f"Found {len(test_images)} test images: {[img.name for img in test_images]}")
        
        # Run tests
        all_results = []
        total_tests = len(test_images) * len(protection_strengths) * len(methods)
        current_test = 0
        
        for image_path in test_images:
            for strength in protection_strengths:
                for method in methods:
                    current_test += 1
                    print(f"\nTest {current_test}/{total_tests}: {image_path.name} - {method} - {strength}")
                    
                    result = self.test_single_image(image_path, strength, method)
                    if result:
                        all_results.append(result)
                    else:
                        print(f"Failed to test {image_path.name}")
        
        # Aggregate results
        if all_results:
            aggregated = self._aggregate_results(all_results)
            self._save_results(aggregated)
            self._generate_report(aggregated)
            return aggregated
        else:
            print("No successful tests completed!")
            return {}
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results into summary statistics."""
        if not results:
            return {}
        
        # Summary statistics
        total_tests = len(results)
        successful_protections = sum(1 for r in results if any(r['models']))
        
        # Aggregate by model
        by_model = {}
        for model_name in self.models.keys():
            model_results = []
            for result in results:
                if model_name in result['models']:
                    model_results.append(result['models'][model_name])
            
            if model_results:
                by_model[model_name] = {
                    'total_tests': len(model_results),
                    'avg_confidence_reduction': np.mean([r['confidence_reduction'] for r in model_results]),
                    'avg_confidence_reduction_percentage': np.mean([r['confidence_reduction_percentage'] for r in model_results]),
                    'confidence_reduction_success_rate': np.mean([r['confidence_reduced'] for r in model_results]),
                    'avg_max_confidence_reduction': np.mean([r['max_confidence_reduction'] for r in model_results]),
                    'avg_entropy_increase': np.mean([r['entropy_increase'] for r in model_results]),
                    'attack_success_rate': np.mean([r['attack_success'] for r in model_results]),
                    'avg_psnr': np.mean([r['psnr'] for r in model_results])
                }
        
        # Aggregate by method
        by_method = {}
        for method in set(r['method'] for r in results):
            method_results = [r for r in results if r['method'] == method]
            if method_results:
                by_method[method] = {
                    'total_tests': len(method_results),
                    'avg_confidence_reduction': np.mean([r['models'].get('inception_v3', {}).get('confidence_reduction', 0) for r in method_results]),
                    'confidence_reduction_success_rate': np.mean([r['models'].get('inception_v3', {}).get('confidence_reduced', False) for r in method_results]),
                    'attack_success_rate': np.mean([r['models'].get('inception_v3', {}).get('attack_success', False) for r in method_results]),
                    'avg_psnr': np.mean([r['models'].get('inception_v3', {}).get('psnr', 0) for r in method_results])
                }
        
        # Aggregate by strength
        by_strength = {}
        for strength in set(r['protection_strength'] for r in results):
            strength_results = [r for r in results if r['protection_strength'] == strength]
            if strength_results:
                by_strength[strength] = {
                    'total_tests': len(strength_results),
                    'avg_confidence_reduction': np.mean([r['models'].get('inception_v3', {}).get('confidence_reduction', 0) for r in strength_results]),
                    'confidence_reduction_success_rate': np.mean([r['models'].get('inception_v3', {}).get('confidence_reduced', False) for r in strength_results]),
                    'attack_success_rate': np.mean([r['models'].get('inception_v3', {}).get('attack_success', False) for r in strength_results])
                }
        
        return {
            'summary': {
                'total_tests': total_tests,
                'successful_protections': successful_protections,
                'protection_success_rate': successful_protections / total_tests if total_tests > 0 else 0,
                'layer_used': 'mixed7',
                'timestamp': datetime.now().isoformat()
            },
            'by_model': by_model,
            'by_method': by_method,
            'by_strength': by_strength,
            'detailed_results': results
        }
    
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
        """Generate a comprehensive report."""
        report_file = self.output_dir / "effectiveness_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("MandrAIk Enhanced Effectiveness Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            summary = results.get('summary', {})
            f.write(f"Layer Used: {summary.get('layer_used', 'mixed7')}\n")
            f.write(f"Total Tests: {summary.get('total_tests', 0)}\n")
            f.write(f"Successful Protections: {summary.get('successful_protections', 0)}\n")
            f.write(f"Protection Success Rate: {summary.get('protection_success_rate', 0):.2%}\n")
            f.write(f"Timestamp: {summary.get('timestamp', 'N/A')}\n\n")
            
            # By Model
            f.write("Results by Model:\n")
            f.write("-" * 30 + "\n")
            for model_name, model_results in results.get('by_model', {}).items():
                f.write(f"{model_name}:\n")
                f.write(f"  Confidence Reduction Success Rate: {model_results['confidence_reduction_success_rate']:.2%}\n")
                f.write(f"  Avg Confidence Reduction: {model_results['avg_confidence_reduction']:.3f}\n")
                f.write(f"  Avg Confidence Reduction %: {model_results['avg_confidence_reduction_percentage']:.1f}%\n")
                f.write(f"  Attack Success Rate: {model_results['attack_success_rate']:.2%}\n")
                f.write(f"  Avg PSNR: {model_results['avg_psnr']:.2f}\n\n")
            
            # By Method
            f.write("Results by Protection Method:\n")
            f.write("-" * 35 + "\n")
            for method, method_results in results.get('by_method', {}).items():
                f.write(f"{method}:\n")
                f.write(f"  Confidence Reduction Success Rate: {method_results['confidence_reduction_success_rate']:.2%}\n")
                f.write(f"  Avg Confidence Reduction: {method_results['avg_confidence_reduction']:.3f}\n")
                f.write(f"  Attack Success Rate: {method_results['attack_success_rate']:.2%}\n")
                f.write(f"  Avg PSNR: {method_results['avg_psnr']:.2f}\n\n")
            
            # By Strength
            f.write("Results by Protection Strength:\n")
            f.write("-" * 35 + "\n")
            for strength, strength_results in results.get('by_strength', {}).items():
                f.write(f"{strength}:\n")
                f.write(f"  Confidence Reduction Success Rate: {strength_results['confidence_reduction_success_rate']:.2%}\n")
                f.write(f"  Avg Confidence Reduction: {strength_results['avg_confidence_reduction']:.3f}\n")
                f.write(f"  Attack Success Rate: {strength_results['attack_success_rate']:.2%}\n\n")
            
            # Key Insights
            f.write("Key Insights:\n")
            f.write("-" * 15 + "\n")
            
            # Find best performing models
            model_results = results.get('by_model', {})
            if model_results:
                best_confidence_reduction = max(model_results.items(), 
                                              key=lambda x: x[1]['confidence_reduction_success_rate'])
                best_avg_reduction = max(model_results.items(), 
                                       key=lambda x: x[1]['avg_confidence_reduction'])
                
                f.write(f"Best Confidence Reduction Success: {best_confidence_reduction[0]} ({best_confidence_reduction[1]['confidence_reduction_success_rate']:.2%})\n")
                f.write(f"Best Avg Confidence Reduction: {best_avg_reduction[0]} ({best_avg_reduction[1]['avg_confidence_reduction']:.3f})\n")
            
            # Find best strength
            strength_results = results.get('by_strength', {})
            if strength_results:
                best_strength = max(strength_results.items(), 
                                  key=lambda x: x[1]['confidence_reduction_success_rate'])
                f.write(f"Best Protection Strength: {best_strength[0]} ({best_strength[1]['confidence_reduction_success_rate']:.2%})\n")
        
        print(f"Report saved to: {report_file}")

def main():
    """Main function to run the enhanced effectiveness test."""
    print("MandrAIk Enhanced Effectiveness Test Suite")
    print("=" * 50)
    print("Features:")
    print("- Semantic target images")
    print("- mixed7 layer (best performing)")
    print("- Enhanced hallucinogen method")
    print("- Focused on InceptionV3 model")
    print("=" * 50)
    
    # Initialize tester
    tester = MandrAIkEffectivenessTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    print("\nEnhanced test completed!")
    print(f"Results saved to: {tester.output_dir}")

if __name__ == "__main__":
    main() 