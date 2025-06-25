#!/usr/bin/env python3
"""
Deep Dream Perturbation Analysis

This script analyzes why deep dream perturbations might increase model confidence
and explores alternative perturbation strategies that could be more effective.
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class DeepDreamAnalysis:
    """Analyze deep dream perturbations and their effects on model confidence."""
    
    def __init__(self):
        self.model = self._load_inception_model()
    
    def _load_inception_model(self):
        """Load InceptionV3 model for analysis."""
        return tf.keras.applications.InceptionV3(
            weights='imagenet',
            include_top=True
        )
    
    def analyze_deep_dream_effect(self, image_path: str) -> Dict:
        """Analyze how deep dream affects model predictions and confidence."""
        
        # Load original image
        original_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
        original_array = tf.keras.preprocessing.image.img_to_array(original_img)
        original_batch = np.expand_dims(original_array, axis=0)
        
        # Preprocess for InceptionV3
        original_processed = tf.keras.applications.inception_v3.preprocess_input(original_batch)
        
        # Get original predictions
        original_preds = self.model.predict(original_processed, verbose=0)
        original_decoded = tf.keras.applications.imagenet_utils.decode_predictions(original_preds, top=5)[0]
        
        # Create deep dream version (simplified)
        dreamed_img = self._create_deep_dream_approximation(original_array)
        dreamed_batch = np.expand_dims(dreamed_img, axis=0)
        dreamed_processed = tf.keras.applications.inception_v3.preprocess_input(dreamed_batch)
        
        # Get dreamed predictions
        dreamed_preds = self.model.predict(dreamed_processed, verbose=0)
        dreamed_decoded = tf.keras.applications.imagenet_utils.decode_predictions(dreamed_preds, top=5)[0]
        
        # Analyze feature activations
        feature_analysis = self._analyze_feature_activations(original_processed, dreamed_processed)
        
        return {
            'original_predictions': original_decoded,
            'dreamed_predictions': dreamed_decoded,
            'feature_analysis': feature_analysis,
            'confidence_change': self._calculate_confidence_change(original_decoded, dreamed_decoded)
        }
    
    def _create_deep_dream_approximation(self, image_array: np.ndarray) -> np.ndarray:
        """Create a simplified deep dream approximation."""
        # This simulates what deep dream does - amplifies features the model recognizes
        
        # Convert to float and normalize
        img = image_array.astype(np.float32) / 255.0
        
        # Apply high-pass filter to enhance edges and features
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply to each channel
        enhanced = np.zeros_like(img)
        for i in range(3):
            enhanced[:,:,i] = cv2.filter2D(img[:,:,i], -1, kernel)
        
        # Add some feature amplification (simulating deep dream)
        enhanced = np.clip(enhanced * 1.2, 0, 1)
        
        # Convert back to 0-255 range
        return (enhanced * 255).astype(np.uint8)
    
    def _analyze_feature_activations(self, original: np.ndarray, dreamed: np.ndarray) -> Dict:
        """Analyze how feature activations change between original and dreamed images."""
        
        # Get intermediate layer activations
        layer_name = 'mixed1'  # Early layer that deep dream targets
        intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        # Get activations
        orig_activations = intermediate_model.predict(original, verbose=0)
        dreamed_activations = intermediate_model.predict(dreamed, verbose=0)
        
        # Calculate activation statistics
        orig_mean = np.mean(orig_activations)
        dreamed_mean = np.mean(dreamed_activations)
        orig_std = np.std(orig_activations)
        dreamed_std = np.std(dreamed_activations)
        
        return {
            'original_mean_activation': float(orig_mean),
            'dreamed_mean_activation': float(dreamed_mean),
            'original_std_activation': float(orig_std),
            'dreamed_std_activation': float(dreamed_std),
            'activation_increase': float(dreamed_mean - orig_mean),
            'activation_amplification': float(dreamed_mean / orig_mean) if orig_mean > 0 else 0
        }
    
    def _calculate_confidence_change(self, original_preds: List, dreamed_preds: List) -> Dict:
        """Calculate how confidence changes between original and dreamed predictions."""
        
        # Get top predictions
        orig_top = original_preds[0]  # (class_id, class_name, confidence)
        dreamed_top = dreamed_preds[0]
        
        # Check if top class changed
        class_changed = orig_top[1] != dreamed_top[1]
        
        # Calculate confidence change
        confidence_change = dreamed_top[2] - orig_top[2]
        
        return {
            'top_class_changed': class_changed,
            'original_class': orig_top[1],
            'dreamed_class': dreamed_top[1],
            'original_confidence': float(orig_top[2]),
            'dreamed_confidence': float(dreamed_top[2]),
            'confidence_change': float(confidence_change),
            'confidence_increased': confidence_change > 0
        }
    
    def explore_alternative_perturbations(self, image_path: str) -> Dict:
        """Explore alternative perturbation strategies that might decrease confidence."""
        
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        alternatives = {
            'noise_perturbation': self._noise_perturbation,
            'blur_perturbation': self._blur_perturbation,
            'compression_perturbation': self._compression_perturbation,
            'geometric_perturbation': self._geometric_perturbation,
            'adversarial_noise': self._adversarial_noise,
            'frequency_perturbation': self._frequency_perturbation
        }
        
        results = {}
        
        for name, func in alternatives.items():
            try:
                perturbed = func(original.copy())
                confidence_effect = self._test_confidence_effect(image_path, perturbed)
                results[name] = confidence_effect
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def _noise_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Add random noise to decrease confidence."""
        noise = np.random.normal(0, 20, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def _blur_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Apply blur to decrease confidence."""
        return cv2.GaussianBlur(image, (15, 15), 5)
    
    def _compression_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Apply JPEG compression artifacts."""
        # Save and reload with compression
        temp_path = "temp_compressed.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 30])
        compressed = cv2.imread(temp_path)
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        import os
        os.remove(temp_path)
        return compressed
    
    def _geometric_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Apply geometric transformations."""
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)  # Small rotation
        return cv2.warpAffine(image, M, (cols, rows))
    
    def _adversarial_noise(self, image: np.ndarray) -> np.ndarray:
        """Add adversarial noise designed to decrease confidence."""
        # Simple adversarial noise pattern
        noise = np.random.normal(0, 15, image.shape)
        # Add some structured noise
        noise += np.sin(np.arange(image.shape[1]) * 0.1)[None, :, None] * 10
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def _frequency_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Perturb frequency domain to affect model features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Add noise to high frequencies
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        
        # Apply noise to high frequencies
        f_shift_filtered = f_shift * mask
        f_shift_filtered += np.random.normal(0, 1000, f_shift.shape) * mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and apply to all channels
        img_back = ((img_back - img_back.min()) / (img_back.max() - img_back.min()) * 255).astype(np.uint8)
        return np.stack([img_back] * 3, axis=-1)
    
    def _test_confidence_effect(self, original_path: str, perturbed_image: np.ndarray) -> Dict:
        """Test how perturbation affects model confidence."""
        
        # Save perturbed image temporarily
        temp_path = "temp_perturbed.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(perturbed_image, cv2.COLOR_RGB2BGR))
        
        # Get predictions for both
        original_img = tf.keras.preprocessing.image.load_img(original_path, target_size=(299, 299))
        original_array = tf.keras.preprocessing.image.img_to_array(original_img)
        original_batch = np.expand_dims(original_array, axis=0)
        original_processed = tf.keras.applications.inception_v3.preprocess_input(original_batch)
        
        perturbed_img = tf.keras.preprocessing.image.load_img(temp_path, target_size=(299, 299))
        perturbed_array = tf.keras.preprocessing.image.img_to_array(perturbed_img)
        perturbed_batch = np.expand_dims(perturbed_array, axis=0)
        perturbed_processed = tf.keras.applications.inception_v3.preprocess_input(perturbed_batch)
        
        # Get predictions
        original_preds = self.model.predict(original_processed, verbose=0)
        perturbed_preds = self.model.predict(perturbed_processed, verbose=0)
        
        original_decoded = tf.keras.applications.imagenet_utils.decode_predictions(original_preds, top=1)[0]
        perturbed_decoded = tf.keras.applications.imagenet_utils.decode_predictions(perturbed_preds, top=1)[0]
        
        # Calculate confidence change
        confidence_change = perturbed_decoded[0][2] - original_decoded[0][2]
        
        # Clean up
        import os
        os.remove(temp_path)
        
        return {
            'original_confidence': float(original_decoded[0][2]),
            'perturbed_confidence': float(perturbed_decoded[0][2]),
            'confidence_change': float(confidence_change),
            'confidence_decreased': confidence_change < 0,
            'prediction_changed': original_decoded[0][1] != perturbed_decoded[0][1]
        }

def main():
    """Run deep dream analysis."""
    analyzer = DeepDreamAnalysis()
    
    # Analyze deep dream effect
    print("Analyzing deep dream perturbation effects...")
    dream_analysis = analyzer.analyze_deep_dream_effect("test_images/eiffel.jpg")
    
    print("\n=== DEEP DREAM ANALYSIS ===")
    print(f"Original top prediction: {dream_analysis['original_predictions'][0][1]} ({dream_analysis['original_predictions'][0][2]:.3f})")
    print(f"Dreamed top prediction: {dream_analysis['dreamed_predictions'][0][1]} ({dream_analysis['dreamed_predictions'][0][2]:.3f})")
    
    confidence_change = dream_analysis['confidence_change']
    print(f"\nConfidence change: {confidence_change['confidence_change']:.3f}")
    print(f"Confidence increased: {confidence_change['confidence_increased']}")
    
    feature_analysis = dream_analysis['feature_analysis']
    print(f"\nFeature activation increase: {feature_analysis['activation_increase']:.3f}")
    print(f"Feature amplification factor: {feature_analysis['activation_amplification']:.2f}x")
    
    # Explore alternatives
    print("\n=== EXPLORING ALTERNATIVE PERTURBATIONS ===")
    alternatives = analyzer.explore_alternative_perturbations("test_images/eiffel.jpg")
    
    for name, result in alternatives.items():
        if 'error' not in result:
            print(f"{name:20s} | Confidence change: {result['confidence_change']:+.3f} | Decreased: {result['confidence_decreased']}")
        else:
            print(f"{name:20s} | Error: {result['error']}")

if __name__ == "__main__":
    main() 