#!/usr/bin/env python3
"""
Alternative Image Spaces and Perturbation Approaches for MandrAIk

This explores different image spaces and perturbation strategies that might be more effective
than the current color-space deep dream approach.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import tensorflow as tf

class AlternativePerturbationSpaces:
    """Explore different image spaces and perturbation strategies."""
    
    def __init__(self):
        self.spaces = {
            'RGB': self._rgb_perturbation,
            'HSV': self._hsv_perturbation,
            'LAB': self._lab_perturbation,
            'YUV': self._yuv_perturbation,
            'YCbCr': self._ycbcr_perturbation,
            'Frequency': self._frequency_perturbation,
            'Gradient': self._gradient_perturbation,
            'Texture': self._texture_perturbation,
            'Edge': self._edge_perturbation,
            'MultiScale': self._multiscale_perturbation
        }
    
    def _rgb_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Direct RGB perturbation - current approach but more aggressive."""
        # Add noise to all channels
        noise = np.random.normal(0, strength * 50, original.shape)
        perturbed = np.clip(original.astype(np.float32) + noise, 0, 255)
        return perturbed.astype(np.uint8)
    
    def _hsv_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """HSV color space perturbation - affects hue, saturation, value."""
        hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Perturb hue (color), saturation, and value
        hsv[:,:,0] = (hsv[:,:,0] + strength * 30) % 180  # Hue is circular
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + strength), 0, 255)  # Saturation
        hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + strength * 0.5), 0, 255)  # Value
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def _lab_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """LAB perturbation - current MandrAIk approach but with L channel modification."""
        lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Perturb all channels, not just A and B
        l, a, b = cv2.split(lab)
        
        # Add structural perturbation to L channel
        l_noise = np.random.normal(0, strength * 20, l.shape)
        l = np.clip(l + l_noise, 0, 255)
        
        # Color perturbations
        a = np.clip(a + strength * 30 * np.random.normal(0, 1, a.shape), 0, 255)
        b = np.clip(b + strength * 30 * np.random.normal(0, 1, b.shape), 0, 255)
        
        perturbed_lab = cv2.merge([l, a, b])
        return cv2.cvtColor(perturbed_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    def _yuv_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """YUV perturbation - affects luminance and chrominance."""
        yuv = cv2.cvtColor(original, cv2.COLOR_RGB2YUV).astype(np.float32)
        
        # Perturb Y (luminance) and UV (chrominance)
        yuv[:,:,0] = np.clip(yuv[:,:,0] + strength * 30, 0, 255)  # Y
        yuv[:,:,1] = np.clip(yuv[:,:,1] + strength * 50, 0, 255)  # U
        yuv[:,:,2] = np.clip(yuv[:,:,2] + strength * 50, 0, 255)  # V
        
        return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2RGB)
    
    def _ycbcr_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """YCbCr perturbation - similar to YUV but different color space."""
        ycbcr = cv2.cvtColor(original, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        
        # Perturb Y (luminance), Cb (blue), Cr (red)
        ycbcr[:,:,0] = np.clip(ycbcr[:,:,0] + strength * 30, 0, 255)  # Y
        ycbcr[:,:,1] = np.clip(ycbcr[:,:,1] + strength * 50, 0, 255)  # Cr
        ycbcr[:,:,2] = np.clip(ycbcr[:,:,2] + strength * 50, 0, 255)  # Cb
        
        return cv2.cvtColor(ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    
    def _frequency_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Frequency domain perturbation - affects image patterns."""
        # Convert to grayscale for frequency analysis
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create frequency mask
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # High-frequency perturbation
        mask = np.ones((rows, cols), np.uint8)
        r = int(30 * strength)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        
        # Apply mask and inverse FFT
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and convert back to RGB
        img_back = ((img_back - img_back.min()) / (img_back.max() - img_back.min()) * 255).astype(np.uint8)
        
        # Apply to all channels
        result = np.stack([img_back] * 3, axis=-1)
        return result
    
    def _gradient_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Gradient-based perturbation - affects edges and contours."""
        # Calculate gradients
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Create gradient perturbation
        grad_perturb = np.sqrt(grad_x**2 + grad_y**2)
        grad_perturb = (grad_perturb / grad_perturb.max() * 255).astype(np.uint8)
        
        # Apply perturbation
        perturbed = original.astype(np.float32) + strength * 50 * np.stack([grad_perturb] * 3, axis=-1) / 255
        return np.clip(perturbed, 0, 255).astype(np.uint8)
    
    def _texture_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Texture-based perturbation - affects local patterns."""
        # Apply Gabor filter to create texture perturbation
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        
        # Create Gabor kernel
        kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        
        # Apply filter
        texture = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # Normalize and apply
        texture = ((texture - texture.min()) / (texture.max() - texture.min()) * 255).astype(np.uint8)
        
        # Apply perturbation
        perturbed = original.astype(np.float32) + strength * 30 * np.stack([texture] * 3, axis=-1) / 255
        return np.clip(perturbed, 0, 255).astype(np.uint8)
    
    def _edge_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Edge-based perturbation - affects object boundaries."""
        # Detect edges
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges for stronger effect
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply edge perturbation
        perturbed = original.astype(np.float32) + strength * 40 * np.stack([edges] * 3, axis=-1) / 255
        return np.clip(perturbed, 0, 255).astype(np.uint8)
    
    def _multiscale_perturbation(self, original: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Multi-scale perturbation - combines multiple approaches."""
        # Apply multiple perturbation types
        perturbed = original.copy()
        
        # 1. Color perturbation
        perturbed = self._hsv_perturbation(perturbed, strength * 0.5)
        
        # 2. Structural perturbation
        perturbed = self._gradient_perturbation(perturbed, strength * 0.3)
        
        # 3. Texture perturbation
        perturbed = self._texture_perturbation(perturbed, strength * 0.2)
        
        return perturbed
    
    def analyze_perturbation_effectiveness(self, image_path: str, strength: float = 0.1) -> Dict:
        """Analyze effectiveness of different perturbation approaches."""
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        results = {}
        
        for space_name, perturbation_func in self.spaces.items():
            try:
                perturbed = perturbation_func(original.copy(), strength)
                
                # Calculate quality metrics
                mse = np.mean((original.astype(np.float32) - perturbed.astype(np.float32)) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
                
                # Calculate structural similarity with proper window size
                try:
                    from skimage.metrics import structural_similarity as ssim
                    # Use smaller window size for small images
                    min_dim = min(original.shape[:2])
                    win_size = min(7, min_dim - 1) if min_dim > 7 else 3
                    if win_size % 2 == 0:  # Ensure odd window size
                        win_size -= 1
                    if win_size < 3:
                        win_size = 3
                    
                    ssim_score = ssim(original, perturbed, multichannel=True, win_size=win_size)
                except Exception as ssim_error:
                    print(f"SSIM calculation failed for {space_name}: {ssim_error}")
                    ssim_score = 0.0
                
                results[space_name] = {
                    'psnr': psnr,
                    'ssim': ssim_score,
                    'perturbed_image': perturbed
                }
                
            except Exception as e:
                print(f"Error with {space_name}: {e}")
                results[space_name] = {'error': str(e)}
        
        return results
    
    def create_comparison_visualization(self, results: Dict, save_path: str = "perturbation_comparison.png"):
        """Create visualization comparing different perturbation approaches."""
        n_spaces = len(results)
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Alternative Perturbation Spaces Comparison', fontsize=16)
        
        for i, (space_name, result) in enumerate(results.items()):
            if i >= 12:  # Limit to 12 subplots
                break
                
            row, col = i // 4, i % 4
            
            if 'error' in result:
                axes[row, col].text(0.5, 0.5, f'Error: {result["error"]}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{space_name}\n(Error)')
            else:
                axes[row, col].imshow(result['perturbed_image'])
                axes[row, col].set_title(f'{space_name}\nPSNR: {result["psnr"]:.1f}dB\nSSIM: {result["ssim"]:.3f}')
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Test alternative perturbation approaches."""
    analyzer = AlternativePerturbationSpaces()
    
    # Test with a sample image
    test_image = "test_images/eiffel.jpg"  # Adjust path as needed
    
    print("Analyzing alternative perturbation approaches...")
    results = analyzer.analyze_perturbation_effectiveness(test_image, strength=0.15)
    
    # Print results
    print("\nPerturbation Effectiveness Analysis:")
    print("=" * 50)
    for space_name, result in results.items():
        if 'error' not in result:
            print(f"{space_name:15s} | PSNR: {result['psnr']:6.1f} dB | SSIM: {result['ssim']:.3f}")
        else:
            print(f"{space_name:15s} | Error: {result['error']}")
    
    # Create visualization
    analyzer.create_comparison_visualization(results)

if __name__ == "__main__":
    main() 