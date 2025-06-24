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
        self._load_model()
    
    def _load_model(self):
        """Load the specified model."""
        if self.model_name == 'inception_v3':
            self.model = tf.keras.applications.InceptionV3(
                weights='imagenet',
                include_top=True
            )
            self.preprocess = tf.keras.applications.inception_v3.preprocess_input
            self.class_names = self._get_imagenet_class_names()
            self.framework = 'tensorflow'
            
        elif self.model_name == 'resnet50':
            self.model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=True
            )
            self.preprocess = tf.keras.applications.resnet50.preprocess_input
            self.class_names = self._get_imagenet_class_names()
            self.framework = 'tensorflow'
            
        elif self.model_name == 'efficientnet_b0':
            self.model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=True
            )
            self.preprocess = tf.keras.applications.efficientnet.preprocess_input
            self.class_names = self._get_imagenet_class_names()
            self.framework = 'tensorflow'
            
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
            
        elif self.model_name == 'pytorch_vit':
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
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
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
    
    def __init__(self, test_images_dir: str = "test_images", output_dir: str = "test_results"):
        self.test_images_dir = Path(test_images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MandrAIk instances with different parameters
        self.mandrAIk_instances = {
            'low': MandrAIk(step_size=0.3, num_ocataves=2),
            'medium': MandrAIk(step_size=0.5, num_ocataves=3),
            'high': MandrAIk(step_size=0.7, num_ocataves=4)
        }
        
        # Modern classification models
        self.classifiers = {
            'InceptionV3': ModernImageClassifier('inception_v3'),
            'ResNet50': ModernImageClassifier('resnet50'),
            'EfficientNetB0': ModernImageClassifier('efficientnet_b0'),
            'PyTorch_ResNet50': ModernImageClassifier('pytorch_resnet50'),
            'PyTorch_EfficientNet': ModernImageClassifier('pytorch_efficientnet'),
            'PyTorch_ViT': ModernImageClassifier('pytorch_vit')
        }
        
        # Test results storage
        self.results = {
            'attack_success_rate': {},
            'confidence_drop': {},
            'top1_accuracy': {},
            'top5_accuracy': {},
            'processing_time': {},
            'image_quality': {}
        }
    
    def load_test_images(self) -> List[Path]:
        """Load test images from directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []
        
        if self.test_images_dir.exists():
            for ext in image_extensions:
                images.extend(self.test_images_dir.glob(f"*{ext}"))
                images.extend(self.test_images_dir.glob(f"*{ext.upper()}"))
        
        if not images:
            print("No test images found. Creating sample images...")
            images = self._create_sample_images()
        
        print(f"Loaded {len(images)} test images")
        return images
    
    def _create_sample_images(self) -> List[Path]:
        """Create realistic sample images for testing."""
        sample_images = []
        
        # Create different types of realistic images
        image_types = [
            ('portrait', (512, 512), (100, 150, 200)),  # Blue-tinted portrait
            ('landscape', (800, 600), (50, 100, 50)),   # Green landscape
            ('artwork', (600, 600), (200, 100, 150)),   # Purple artwork
            ('photography', (640, 480), (150, 150, 150)) # Gray photography
        ]
        
        for img_type, size, color in image_types:
            img_path = self.test_images_dir / f"sample_{img_type}.jpg"
            
            # Create realistic image with texture
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            
            # Add some structure based on type
            if img_type == 'portrait':
                # Add face-like structure
                center_y, center_x = size[0]//2, size[1]//2
                for y in range(size[0]):
                    for x in range(size[1]):
                        dist = np.sqrt((y-center_y)**2 + (x-center_x)**2)
                        if dist < 100:
                            img[y, x] = [200, 180, 160]  # Skin tone
                        else:
                            img[y, x] = color
            
            elif img_type == 'landscape':
                # Add horizon line
                horizon = size[0] // 2
                img[:horizon] = [100, 150, 255]  # Sky
                img[horizon:] = [34, 139, 34]    # Ground
            
            # Add some noise for realism
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            Image.fromarray(img).save(img_path)
            sample_images.append(img_path)
        
        return sample_images
    
    def calculate_image_quality(self, original_path: Path, protected_path: Path) -> Dict[str, float]:
        """Calculate image quality metrics."""
        try:
            original = cv2.imread(str(original_path))
            protected = cv2.imread(str(protected_path))
            
            if original is None or protected is None:
                return {'psnr': 0.0, 'ssim': 0.0}
            
            # Ensure same size
            if original.shape != protected.shape:
                protected = cv2.resize(protected, (original.shape[1], original.shape[0]))
            
            # Calculate PSNR
            mse = np.mean((original.astype(np.float64) - protected.astype(np.float64)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
            
            # Calculate SSIM (simplified)
            mu1 = np.mean(original)
            mu2 = np.mean(protected)
            sigma1 = np.std(original)
            sigma2 = np.std(protected)
            sigma12 = np.mean((original - mu1) * (protected - mu2))
            
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
            
            return {'psnr': float(psnr), 'ssim': float(ssim)}
            
        except Exception as e:
            print(f"Error calculating image quality: {e}")
            return {'psnr': 0.0, 'ssim': 0.0}
    
    def test_single_image(self, image_path: Path, protection_strength: str = 'medium') -> Dict:
        """Test effectiveness on a single image."""
        results = {}
        
        # Get original predictions
        original_predictions = {}
        for model_name, classifier in self.classifiers.items():
            original_predictions[model_name] = classifier.predict(str(image_path))
        
        # Apply MandrAIk protection
        mandrAIk = self.mandrAIk_instances[protection_strength]
        protected_path = self.output_dir / f"protected_{image_path.stem}_{protection_strength}.jpg"
        
        start_time = time.time()
        try:
            # Use different protection strengths for the poison method
            protection_strengths = {'low': 0.1, 'medium': 0.15, 'high': 0.2}
            mandrAIk.poison(str(image_path), str(protected_path), protection_strengths[protection_strength])
            processing_time = time.time() - start_time
        except Exception as e:
            print(f"Error protecting {image_path}: {e}")
            return {'error': str(e)}
        
        # Get protected predictions
        protected_predictions = {}
        for model_name, classifier in self.classifiers.items():
            protected_predictions[model_name] = classifier.predict(str(protected_path))
        
        # Calculate metrics
        for model_name in self.classifiers.keys():
            orig_preds = original_predictions[model_name]
            prot_preds = protected_predictions[model_name]
            
            if not orig_preds or not prot_preds:
                continue
            
            # Attack success rate (top-1 class changed)
            attack_success = orig_preds[0][0] != prot_preds[0][0]
            
            # Confidence drop
            confidence_drop = orig_preds[0][1] - prot_preds[0][1]
            
            # Top-1 accuracy (original correct prediction maintained)
            top1_accuracy = orig_preds[0][0] == prot_preds[0][0]
            
            # Top-5 accuracy (original prediction in top-5)
            top5_accuracy = any(orig_preds[0][0] == pred[0] for pred in prot_preds[:5])
            
            results[model_name] = {
                'attack_success': attack_success,
                'confidence_drop': confidence_drop,
                'top1_accuracy': top1_accuracy,
                'top5_accuracy': top5_accuracy,
                'original_top1': orig_preds[0],
                'protected_top1': prot_preds[0],
                'original_top5': orig_preds[:5],
                'protected_top5': prot_preds[:5]
            }
        
        # Image quality metrics
        quality_metrics = self.calculate_image_quality(image_path, protected_path)
        results['quality'] = quality_metrics
        results['processing_time'] = processing_time
        
        return results
    
    def run_comprehensive_test(self, protection_strengths: List[str] = None) -> Dict:
        """Run comprehensive effectiveness test."""
        if protection_strengths is None:
            protection_strengths = ['low', 'medium', 'high']
        
        print("Loading test images...")
        test_images = self.load_test_images()
        
        if not test_images:
            print("No test images available!")
            return {}
        
        print(f"Running comprehensive test on {len(test_images)} images...")
        
        all_results = {}
        
        for strength in protection_strengths:
            print(f"\nTesting protection strength: {strength}")
            strength_results = []
            
            for i, image_path in enumerate(test_images):
                print(f"  Processing image {i+1}/{len(test_images)}: {image_path.name}")
                result = self.test_single_image(image_path, strength)
                
                if 'error' not in result:
                    strength_results.append(result)
                else:
                    print(f"    Error: {result['error']}")
            
            if strength_results:
                all_results[strength] = self._aggregate_results(strength_results)
        
        # Save results
        self._save_results(all_results)
        self._generate_report(all_results)
        
        return all_results
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across all images."""
        aggregated = {}
        
        for model_name in self.classifiers.keys():
            model_results = [r.get(model_name, {}) for r in results if model_name in r]
            
            if not model_results:
                continue
            
            # Calculate statistics
            attack_success_rate = np.mean([r.get('attack_success', False) for r in model_results])
            confidence_drops = [r.get('confidence_drop', 0) for r in model_results]
            top1_accuracies = [r.get('top1_accuracy', False) for r in model_results]
            top5_accuracies = [r.get('top5_accuracy', False) for r in model_results]
            
            aggregated[model_name] = {
                'attack_success_rate': float(attack_success_rate),
                'avg_confidence_drop': float(np.mean(confidence_drops)),
                'std_confidence_drop': float(np.std(confidence_drops)),
                'top1_accuracy': float(np.mean(top1_accuracies)),
                'top5_accuracy': float(np.mean(top5_accuracies)),
                'num_images': len(model_results)
            }
        
        # Aggregate quality metrics
        quality_metrics = [r.get('quality', {}) for r in results if 'quality' in r]
        if quality_metrics:
            psnr_values = [q.get('psnr', 0) for q in quality_metrics]
            ssim_values = [q.get('ssim', 0) for q in quality_metrics]
            
            aggregated['quality'] = {
                'avg_psnr': float(np.mean(psnr_values)),
                'std_psnr': float(np.std(psnr_values)),
                'avg_ssim': float(np.mean(ssim_values)),
                'std_ssim': float(np.std(ssim_values))
            }
        
        # Aggregate processing time
        processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
        if processing_times:
            aggregated['processing_time'] = {
                'avg_time': float(np.mean(processing_times)),
                'std_time': float(np.std(processing_times)),
                'total_time': float(np.sum(processing_times))
            }
        
        return aggregated
    
    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        output_file = self.output_dir / "effectiveness_results.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_converted = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def _generate_report(self, results: Dict):
        """Generate comprehensive report with visualizations."""
        report_file = self.output_dir / "effectiveness_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# MandrAIk Effectiveness Test Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for strength, strength_results in results.items():
                f.write(f"## Protection Strength: {strength.upper()}\n\n")
                
                # Model performance table
                f.write("### Model Performance\n\n")
                f.write("| Model | Attack Success Rate | Avg Confidence Drop | Top-1 Accuracy | Top-5 Accuracy |\n")
                f.write("|-------|-------------------|-------------------|----------------|----------------|\n")
                
                for model_name, model_results in strength_results.items():
                    if model_name in ['quality', 'processing_time']:
                        continue
                    
                    f.write(f"| {model_name} | {model_results['attack_success_rate']:.3f} | "
                           f"{model_results['avg_confidence_drop']:.3f} | "
                           f"{model_results['top1_accuracy']:.3f} | "
                           f"{model_results['top5_accuracy']:.3f} |\n")
                
                f.write("\n")
                
                # Quality metrics
                if 'quality' in strength_results:
                    quality = strength_results['quality']
                    f.write("### Image Quality Metrics\n\n")
                    f.write(f"- Average PSNR: {quality['avg_psnr']:.2f} dB\n")
                    f.write(f"- Average SSIM: {quality['avg_ssim']:.3f}\n\n")
                
                # Processing time
                if 'processing_time' in strength_results:
                    proc_time = strength_results['processing_time']
                    f.write("### Processing Performance\n\n")
                    f.write(f"- Average processing time: {proc_time['avg_time']:.2f} seconds\n")
                    f.write(f"- Total processing time: {proc_time['total_time']:.2f} seconds\n\n")
        
        print(f"Report generated: {report_file}")
        
        # Create visualizations
        self._create_visualizations(results)
    
    def _create_visualizations(self, results: Dict):
        """Create visualization plots."""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MandrAIk Effectiveness Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        strengths = list(results.keys())
        models = list(results[strengths[0]].keys())
        models = [m for m in models if m not in ['quality', 'processing_time']]
        
        # 1. Attack Success Rate
        ax1 = axes[0, 0]
        attack_rates = []
        for strength in strengths:
            rates = [results[strength][model]['attack_success_rate'] for model in models]
            attack_rates.append(rates)
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, strength in enumerate(strengths):
            ax1.bar(x + i*width, attack_rates[i], width, label=strength.capitalize(), alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Attack Success Rate')
        ax1.set_title('Attack Success Rate by Model and Protection Strength')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence Drop
        ax2 = axes[0, 1]
        confidence_drops = []
        for strength in strengths:
            drops = [results[strength][model]['avg_confidence_drop'] for model in models]
            confidence_drops.append(drops)
        
        for i, strength in enumerate(strengths):
            ax2.bar(x + i*width, confidence_drops[i], width, label=strength.capitalize(), alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Average Confidence Drop')
        ax2.set_title('Confidence Drop by Model and Protection Strength')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Image Quality (PSNR)
        ax3 = axes[1, 0]
        psnr_values = []
        for strength in strengths:
            if 'quality' in results[strength]:
                psnr_values.append(results[strength]['quality']['avg_psnr'])
            else:
                psnr_values.append(0)
        
        ax3.bar(strengths, psnr_values, alpha=0.8, color='green')
        ax3.set_xlabel('Protection Strength')
        ax3.set_ylabel('Average PSNR (dB)')
        ax3.set_title('Image Quality (PSNR) by Protection Strength')
        ax3.grid(True, alpha=0.3)
        
        # 4. Processing Time
        ax4 = axes[1, 1]
        proc_times = []
        for strength in strengths:
            if 'processing_time' in results[strength]:
                proc_times.append(results[strength]['processing_time']['avg_time'])
            else:
                proc_times.append(0)
        
        ax4.bar(strengths, proc_times, alpha=0.8, color='orange')
        ax4.set_xlabel('Protection Strength')
        ax4.set_ylabel('Average Processing Time (seconds)')
        ax4.set_title('Processing Time by Protection Strength')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "effectiveness_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {plot_file}")

def main():
    """Main function to run the effectiveness test."""
    print("MandrAIk Effectiveness Test Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = MandrAIkEffectivenessTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    if results:
        print("\n" + "=" * 50)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Results saved to: {tester.output_dir}")
        print("Files generated:")
        print(f"  - effectiveness_results.json (raw data)")
        print(f"  - effectiveness_report.md (detailed report)")
        print(f"  - effectiveness_analysis.png (visualizations)")
        print(f"  - protected_*.jpg (protected test images)")
    else:
        print("Test failed to complete!")

if __name__ == "__main__":
    main() 