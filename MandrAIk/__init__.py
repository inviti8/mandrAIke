import os
from pathlib import Path
import click
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import random
import scipy.fftpack as fftpack
import json
from scipy.fft import fft2, ifft2

FILE_PATH = Path(__file__).parent

class MandrAIk:
    def __init__(self, model_name='InceptionV3', steps=3, step_size=0.8, num_ocataves=3, octave_scale=2.5, noise_level="Min", layer_name='mixed7', max_dim=512):
        """
        Initialize the image protection system.
        
        Args:
            model_name: Pre-trained model to use for protection
            layer_name: Layer to target for perturbation
            steps: Iterations in the dream
            step_size: 0.0 to 1, increases intensity of dreamimage
            num_ocataves: Number of octaves
            noise_level: default noise added to image
            octave_scale: Scales dream image elements
            max_dim: max dimensions of image
        """
        self.model_name = model_name
        self.steps = steps
        self.step_size = step_size
        self.num_ocataves = num_ocataves
        self.octave_scale = octave_scale
        self.noise_level = noise_level
        self.layer_name = layer_name
        self.max_dim = max_dim
        self.model = self._load_model()
        self.noise_img = os.path.join(FILE_PATH, 'noised_image.jpg')
        
        # Store dreamed target for reuse
        self.dreamed_target = None
        
    def _load_model(self):
        """Load pre-trained model."""
        if self.model_name == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(
                include_top=False, 
                weights='imagenet'
            )
            target_layer = base_model.get_layer(self.layer_name).output
            return tf.keras.Model(inputs=base_model.input, outputs=target_layer)
        # Add other models as needed

    def _load_image(self, path):
        """Load and preprocess image with support for multiple formats (JPG, PNG, etc.)."""
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((299, 299))
            img_array = np.array(img).astype(np.float32)
            return tf.keras.applications.inception_v3.preprocess_input(img_array)
        except Exception as e:
            raise ValueError(f"Failed to load image {path}: {e}. Supported formats: JPG, PNG, JPEG, BMP, TIFF")
    
    def _dream_target_image(self, target_path: str) -> np.ndarray:
        """Dream the target image to get its amplified features."""
        print(f"Dreaming target image: {target_path}")
        
        target_img = self._load_image(target_path)
        
        # Run deep dream on the target image
        dreamed_target = self._run_deep_dream(
            img=target_img,
            steps_per_octave=self.steps,
            step_size=self.step_size,
            octaves=self.num_ocataves,
            octave_scale=self.octave_scale
        )
        
        # Convert to RGB
        dreamed_rgb = self._deprocess(dreamed_target.numpy())
        return dreamed_rgb
    
    def _run_deep_dream(self, img: tf.Tensor, steps_per_octave: int, 
                       step_size: float, octaves: int, octave_scale: float) -> tf.Tensor:
        """Run deep dream on the target image."""
        base_shape = tf.shape(img)[:-1]
        img = tf.identity(img)
        
        for octave in range(octaves):
            new_size = tf.cast(tf.cast(base_shape, tf.float32) * (octave_scale ** octave), tf.int32)
            img = tf.image.resize(img, new_size)
            
            for step in range(steps_per_octave):
                img, loss = self._deepdream_step(img, self.model, step_size)
                if step % 10 == 0:
                    print(f"  Octave {octave+1}/{octaves}, Step {step}, Loss: {loss.numpy():.4f}")
        
        return img
    
    def _generate_dreamed_target(self, target_shape=None):
        """Generate dreamed target from random noise at the target resolution."""
        print("No target image provided. Generating dreamed target from random noise...")
        
        # Use target_shape if provided, otherwise default to 299x299
        if target_shape is None:
            target_shape = (299, 299, 3)
        
        # Create random noise as target at the correct resolution
        random_target = tf.random.normal((1, target_shape[0], target_shape[1], 3), mean=0.0, stddev=0.1)
        random_target = tf.squeeze(random_target, axis=0)
        
        # Dream the random noise
        dreamed_random = self._run_deep_dream(
            img=random_target,
            steps_per_octave=self.steps,
            step_size=self.step_size,
            octaves=self.num_ocataves,
            octave_scale=self.octave_scale
        )
        
        self.dreamed_target = self._deprocess(dreamed_random.numpy())
    
    
    def _apply_dreamed_perturbation(self, original_img: np.ndarray, dreamed_target: np.ndarray, protection_strength: float) -> np.ndarray:
        """
        Apply perturbation to original image based on dreamed target features.
        
        Args:
            original_img: Original image (RGB, 0-255)
            dreamed_target: Dreamed target image (RGB, 0-255)
            protection_strength: Strength of perturbation (0.0-1.0)
        
        Returns:
            Perturbed image
        """
        # Ensure both images are the same size
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        
        # Convert to LAB color space for better perturbation control
        original_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
        dreamed_lab = cv2.cvtColor(dreamed_target, cv2.COLOR_RGB2LAB)
        
        # Split channels
        orig_l, orig_a, orig_b = cv2.split(original_lab)
        dream_l, dream_a, dream_b = cv2.split(dreamed_lab)
        
        # Calculate differences (dreamed - original)
        l_diff = dream_l.astype(np.float32) - orig_l.astype(np.float32)
        a_diff = dream_a.astype(np.float32) - orig_a.astype(np.float32)
        b_diff = dream_b.astype(np.float32) - orig_b.astype(np.float32)
        
        # Apply perturbation based on dreamed differences
        # Use dreamed features to perturb the original
        new_l = np.clip(orig_l.astype(np.float32) + (l_diff * protection_strength), 0, 255).astype(np.uint8)
        new_a = np.clip(orig_a.astype(np.float32) + (a_diff * protection_strength), 0, 255).astype(np.uint8)
        new_b = np.clip(orig_b.astype(np.float32) + (b_diff * protection_strength), 0, 255).astype(np.uint8)
        
        # Merge channels and convert back to RGB
        perturbed_lab = cv2.merge([new_l, new_a, new_b])
        perturbed_rgb = cv2.cvtColor(perturbed_lab, cv2.COLOR_LAB2RGB)
        
        return perturbed_rgb
    
    def _apply_dreamed_perturbation_lab(self, original_img: np.ndarray, dreamed_target: np.ndarray, protection_strength: float) -> np.ndarray:
        """
        Apply perturbation in LAB color space.
        """
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        original_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
        dreamed_lab = cv2.cvtColor(dreamed_target, cv2.COLOR_RGB2LAB)
        orig_l, orig_a, orig_b = cv2.split(original_lab)
        dream_l, dream_a, dream_b = cv2.split(dreamed_lab)
        l_diff = dream_l.astype(np.float32) - orig_l.astype(np.float32)
        a_diff = dream_a.astype(np.float32) - orig_a.astype(np.float32)
        b_diff = dream_b.astype(np.float32) - orig_b.astype(np.float32)
        new_l = np.clip(orig_l.astype(np.float32) + (l_diff * protection_strength), 0, 255).astype(np.uint8)
        new_a = np.clip(orig_a.astype(np.float32) + (a_diff * protection_strength), 0, 255).astype(np.uint8)
        new_b = np.clip(orig_b.astype(np.float32) + (b_diff * protection_strength), 0, 255).astype(np.uint8)
        perturbed_lab = cv2.merge([new_l, new_a, new_b])
        perturbed_rgb = cv2.cvtColor(perturbed_lab, cv2.COLOR_LAB2RGB)
        return perturbed_rgb

    def _apply_dreamed_perturbation_rgb(self, original_img: np.ndarray, dreamed_target: np.ndarray, protection_strength: float) -> np.ndarray:
        """
        Apply perturbation in RGB space.
        """
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        original_float = original_img.astype(np.float32)
        dreamed_float = dreamed_target.astype(np.float32)
        diff = dreamed_float - original_float
        perturbed = np.clip(original_float + diff * protection_strength, 0, 255).astype(np.uint8)
        return perturbed

    def _apply_dreamed_perturbation_hsv(self, original_img: np.ndarray, dreamed_target: np.ndarray, protection_strength: float) -> np.ndarray:
        """
        Apply perturbation in HSV color space.
        """
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        original_hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        dreamed_hsv = cv2.cvtColor(dreamed_target, cv2.COLOR_RGB2HSV)
        orig_h, orig_s, orig_v = cv2.split(original_hsv)
        dream_h, dream_s, dream_v = cv2.split(dreamed_hsv)
        h_diff = dream_h.astype(np.float32) - orig_h.astype(np.float32)
        s_diff = dream_s.astype(np.float32) - orig_s.astype(np.float32)
        v_diff = dream_v.astype(np.float32) - orig_v.astype(np.float32)
        new_h = np.clip(orig_h.astype(np.float32) + (h_diff * protection_strength), 0, 255).astype(np.uint8)
        new_s = np.clip(orig_s.astype(np.float32) + (s_diff * protection_strength), 0, 255).astype(np.uint8)
        new_v = np.clip(orig_v.astype(np.float32) + (v_diff * protection_strength), 0, 255).astype(np.uint8)
        perturbed_hsv = cv2.merge([new_h, new_s, new_v])
        perturbed_rgb = cv2.cvtColor(perturbed_hsv, cv2.COLOR_HSV2RGB)
        return perturbed_rgb

    def _apply_dreamed_perturbation_yuv(self, original_img: np.ndarray, dreamed_target: np.ndarray, protection_strength: float) -> np.ndarray:
        """
        Apply perturbation in YUV color space.
        """
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        original_yuv = cv2.cvtColor(original_img, cv2.COLOR_RGB2YUV)
        dreamed_yuv = cv2.cvtColor(dreamed_target, cv2.COLOR_RGB2YUV)
        orig_y, orig_u, orig_v = cv2.split(original_yuv)
        dream_y, dream_u, dream_v = cv2.split(dreamed_yuv)
        y_diff = dream_y.astype(np.float32) - orig_y.astype(np.float32)
        u_diff = dream_u.astype(np.float32) - orig_u.astype(np.float32)
        v_diff = dream_v.astype(np.float32) - orig_v.astype(np.float32)
        new_y = np.clip(orig_y.astype(np.float32) + (y_diff * protection_strength), 0, 255).astype(np.uint8)
        new_u = np.clip(orig_u.astype(np.float32) + (u_diff * protection_strength), 0, 255).astype(np.uint8)
        new_v = np.clip(orig_v.astype(np.float32) + (v_diff * protection_strength), 0, 255).astype(np.uint8)
        perturbed_yuv = cv2.merge([new_y, new_u, new_v])
        perturbed_rgb = cv2.cvtColor(perturbed_yuv, cv2.COLOR_YUV2RGB)
        return perturbed_rgb
    
    def dream(self, image_path, output_path=None):
        """
        Get dream output based on Perlin noise generation.
        
        Args:
            image_path: Path to input image (used for dimensions)
            output_path: Path to save dreamed image
        """
        print("Generating dreamed image from Perlin noise...")
        
        # Generate Perlin noise target and dream it (similar to poison method)
        dreamed_img = self._generate_noise_target(image_path)
        
        if output_path:
            self._save_image(dreamed_img, output_path)
        
        return dreamed_img
    
    def _noised_image(self, img):
        """Add noise to image."""
        noise_levels = {"Min": 0.1, "Medium": 0.3, "Max": 0.5}
        noise = tf.random.normal(img.shape, mean=0.0, stddev=noise_levels[self.noise_level])
        return img + noise
    
    def _apply_watermark(self, image, msg="ALL RIGHTS RESERVED"):
        """Apply watermark to image."""
        # Convert to PIL for text overlay
        pil_img = Image.fromarray(image.astype(np.uint8))
        
        # Add text watermark
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_img)
        
        # Try to use a default font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Get text size and position it
        bbox = draw.textbbox((0, 0), msg, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position in bottom right with padding
        x = image.shape[1] - text_width - 10
        y = image.shape[0] - text_height - 10
        
        # Draw text with outline for visibility
        draw.text((x+1, y+1), msg, fill=(0, 0, 0), font=font)  # Black outline
        draw.text((x, y), msg, fill=(255, 255, 255), font=font)  # White text
        
        return np.array(pil_img)

    def _calc_loss(self, img, model):
        """Calculate loss for deep dream."""
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = model(img_batch)
        return tf.reduce_mean(layer_activations)
    
    @tf.function
    def _deepdream_step(self, img, model, step_size):
        """Single step of deep dream."""
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self._calc_loss(img, model)
        
        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        img = img + gradients * step_size
        img = tf.clip_by_value(img, -1, 1)
        
        return img, loss
    
    def _deprocess(self, img):
        """Convert from model format back to RGB."""
        img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def _save_image(self, img, path):
        """Save image to path, preserving format based on file extension."""
        try:
            # Determine format from file extension
            ext = os.path.splitext(path)[1].lower()
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Save with appropriate format
            if ext in ['.png', '.PNG']:
                cv2.imwrite(path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            elif ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                cv2.imwrite(path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                # Default to PNG for other formats
                cv2.imwrite(path, img_bgr)
                
        except Exception as e:
            raise ValueError(f"Failed to save image to {path}: {e}")
    
    
    def _apply_strategic_noise(self, image: np.ndarray, noise_strength: float, step: int) -> np.ndarray:
        """
        Apply strategic noise based on step position.
        
        Args:
            image: Input image
            noise_strength: Strength of noise
            step: Current step number
        
        Returns:
            Image with strategic noise
        """
        # Different noise patterns for different steps
        if step < 2:
            # Early steps: high-frequency noise for detail disruption
            noise = np.random.normal(0, noise_strength * 25, image.shape)
        elif step < 4:
            # Middle steps: medium-frequency noise
            noise = np.random.normal(0, noise_strength * 20, image.shape)
        else:
            # Final steps: low-frequency noise for smooth finish
            noise = np.random.normal(0, noise_strength * 15, image.shape)
        
        # Apply noise with clipping
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def _apply_final_enhancement(self, image: np.ndarray, protection_strength: float) -> np.ndarray:
        """
        Apply final enhancement to maximize adversarial effect.
        
        Args:
            image: Input image
            protection_strength: Overall protection strength
        
        Returns:
            Enhanced image
        """
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Apply subtle edge enhancement to create adversarial patterns
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * (protection_strength * 0.1)
        
        enhanced = cv2.filter2D(img_float, -1, kernel)
        
        # Blend with original to maintain quality
        final_image = img_float * 0.8 + enhanced * 0.2
        
        return np.clip(final_image, 0, 255).astype(np.uint8)

    def _targeted_fgsm_perturbation(self, original_img: np.ndarray, dreamed_target: np.ndarray, epsilon: float = 16/255) -> np.ndarray:
        """
        Apply targeted FGSM perturbation to move image toward dreamed target with spatial flipping for enhanced confusion.
        
        Args:
            original_img: Original image (RGB, 0-255)
            dreamed_target: Dreamed target image (RGB, 0-255)
            epsilon: Perturbation magnitude (default: 16/255)
        
        Returns:
            Perturbed image
        """
        
        # Resize dreamed target to match original if needed
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        
        # Create flipped versions of dreamed target for additional confusion
        dreamed_target_flipped_x = cv2.flip(dreamed_target, 1)  # Flip horizontally
        dreamed_target_flipped_y = cv2.flip(dreamed_target, 0)  # Flip vertically
        dreamed_target_flipped_xy = cv2.flip(dreamed_target, -1)  # Flip both
        
        # Convert to float32 and normalize to [0, 1]
        original_float = original_img.astype(np.float32) / 255.0
        dreamed_float = dreamed_target.astype(np.float32) / 255.0
        dreamed_flipped_x_float = dreamed_target_flipped_x.astype(np.float32) / 255.0
        dreamed_flipped_y_float = dreamed_target_flipped_y.astype(np.float32) / 255.0
        dreamed_flipped_xy_float = dreamed_target_flipped_xy.astype(np.float32) / 255.0
        
        # Convert to tensors
        original_tensor = tf.convert_to_tensor(original_float)
        dreamed_tensor = tf.convert_to_tensor(dreamed_float)
        dreamed_flipped_x_tensor = tf.convert_to_tensor(dreamed_flipped_x_float)
        dreamed_flipped_y_tensor = tf.convert_to_tensor(dreamed_flipped_y_float)
        dreamed_flipped_xy_tensor = tf.convert_to_tensor(dreamed_flipped_xy_float)
        
        # Add batch dimension
        original_tensor = tf.expand_dims(original_tensor, 0)
        dreamed_tensor = tf.expand_dims(dreamed_tensor, 0)
        dreamed_flipped_x_tensor = tf.expand_dims(dreamed_flipped_x_tensor, 0)
        dreamed_flipped_y_tensor = tf.expand_dims(dreamed_flipped_y_tensor, 0)
        dreamed_flipped_xy_tensor = tf.expand_dims(dreamed_flipped_xy_tensor, 0)
        
        # Preprocess for model
        original_input = tf.keras.applications.inception_v3.preprocess_input(original_tensor * 255.0)
        dreamed_input = tf.keras.applications.inception_v3.preprocess_input(dreamed_tensor * 255.0)
        dreamed_flipped_x_input = tf.keras.applications.inception_v3.preprocess_input(dreamed_flipped_x_tensor * 255.0)
        dreamed_flipped_y_input = tf.keras.applications.inception_v3.preprocess_input(dreamed_flipped_y_tensor * 255.0)
        dreamed_flipped_xy_input = tf.keras.applications.inception_v3.preprocess_input(dreamed_flipped_xy_tensor * 255.0)
        
        # Compute gradients with respect to input
        with tf.GradientTape() as tape:
            tape.watch(original_input)
            
            # Get predictions for current input
            current_preds = self.model(original_input)
            dreamed_preds = self.model(dreamed_input)
            dreamed_flipped_x_preds = self.model(dreamed_flipped_x_input)
            dreamed_flipped_y_preds = self.model(dreamed_flipped_y_input)
            dreamed_flipped_xy_preds = self.model(dreamed_flipped_xy_input)
            
            # Combine losses from original and flipped targets for enhanced confusion
            loss_original = tf.reduce_mean(tf.square(current_preds - dreamed_preds))
            loss_flipped_x = tf.reduce_mean(tf.square(current_preds - dreamed_flipped_x_preds))
            loss_flipped_y = tf.reduce_mean(tf.square(current_preds - dreamed_flipped_y_preds))
            loss_flipped_xy = tf.reduce_mean(tf.square(current_preds - dreamed_flipped_xy_preds))
            
            # Weighted combination of all losses
            targeted_loss = (0.4 * loss_original + 
                           0.2 * loss_flipped_x + 
                           0.2 * loss_flipped_y + 
                           0.2 * loss_flipped_xy)
        
        # Compute gradients
        gradients = tape.gradient(targeted_loss, original_input)
        
        # Apply FGSM perturbation: move in direction of gradient
        perturbation = epsilon * tf.sign(gradients)
        perturbed_input = original_input + perturbation
        
        # Clip to valid range
        perturbed_input = tf.clip_by_value(perturbed_input, -1.0, 1.0)
        
        # Convert back to [0, 1] range and remove batch dimension
        perturbed_tensor = (perturbed_input + 1.0) / 2.0
        perturbed_tensor = perturbed_tensor[0]
        
        # Convert back to uint8
        perturbed_img = (perturbed_tensor.numpy() * 255).astype(np.uint8)
        
        return perturbed_img

    def _create_perlin_noise(self, width: int = 299, height: int = 299) -> tf.Tensor:
        """
        Create Perlin-like noise for natural patterns optimized for deep dream.
        
        Args:
            width: Width of the noise image (default: 299)
            height: Height of the noise image (default: 299)
            
        Returns:
            Generated Perlin noise tensor of specified dimensions
        """
        # Simplified Perlin noise approximation optimized for deep dream
        base_noise = tf.random.normal((1, height, width, 3), mean=0.0, stddev=0.03)
        
        # Create multiple octaves of noise for natural patterns
        noise_sum = base_noise[0]
        amplitude = 1.0
        frequency = 1.0
        
        for i in range(4):  # 4 octaves for rich detail
            # Create noise at different frequencies
            freq_noise = tf.random.normal((1, height, width, 3), mean=0.0, stddev=0.03 * amplitude)
            # Apply frequency scaling (simplified)
            freq_noise = tf.image.resize(freq_noise, (height, width))
            noise_sum += freq_noise[0] * amplitude
            
            amplitude *= 0.5
            frequency *= 2.0
        
        return noise_sum

    def _generate_noise_target(self, image_path: str) -> np.ndarray:
        """
        Generate a noise target using Perlin noise for protection.
        Args:
            image_path: Path to the original image (for size reference)
        Returns:
            Dreamed noise target as numpy array
        """
        # Load original image to get dimensions
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Failed to load image {image_path}")
        height, width = original_img.shape[:2]
        # Generate dreamed target at the original image's resolution
        self._generate_dreamed_target(target_shape=(height, width, 3))
        return self.dreamed_target

    def _compute_gradients(self, image: np.ndarray, model) -> np.ndarray:
        """
        Compute gradients of the model output with respect to the input image.
        
        Args:
            image: Input image (preprocessed for model)
            model: TensorFlow model
        
        Returns:
            Gradient tensor
        """
        with tf.GradientTape() as tape:
            tape.watch(image)
            output = model(image)
            # Use the mean activation as loss
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, image)
        return gradients

    def _create_dream_based_mask(self, original_img: np.ndarray, dreamed_target: np.ndarray) -> np.ndarray:
        """
        Create perturbation mask based on dream target patterns.
        
        Args:
            original_img: Original image (RGB, 0-255)
            dreamed_target: Dreamed target image (RGB, 0-255)
        
        Returns:
            Mask for applying perturbations (0.0-1.0)
        """
        # Ensure same size
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        
        # Convert to grayscale for pattern analysis
        dream_gray = cv2.cvtColor(dreamed_target, cv2.COLOR_RGB2GRAY)
        orig_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        
        # Compute difference to find dream patterns
        pattern_diff = np.abs(dream_gray.astype(np.float32) - orig_gray.astype(np.float32))
        
        # Normalize pattern differences
        pattern_mask = pattern_diff / (pattern_diff.max() + 1e-8)
        
        # Apply more aggressive Gaussian smoothing for natural transitions
        pattern_mask = cv2.GaussianBlur(pattern_mask, (21, 21), 5)
        
        # Create more aggressive threshold - focus on top 20% of patterns
        threshold = np.percentile(pattern_mask, 80)  # Top 20% of patterns get strong perturbation
        
        # Create mask with more aggressive transitions
        mask = np.where(pattern_mask > threshold, 1.0, 0.1)  # Strong vs weak perturbation
        
        # Add high-frequency emphasis using Laplacian
        laplacian = cv2.Laplacian(orig_gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian = laplacian / (laplacian.max() + 1e-8)
        
        # Combine pattern mask with high-frequency emphasis
        combined_mask = mask * 0.7 + laplacian * 0.3
        combined_mask = np.clip(combined_mask, 0.1, 1.0)
        
        # Ensure mask has same dimensions as image
        combined_mask = np.stack([combined_mask] * 3, axis=-1)  # Apply to all channels
        
        return combined_mask

    def _apply_masked_gradient_perturbation(self, original_img: np.ndarray, dreamed_target: np.ndarray, 
                                          protection_strength: float, model) -> np.ndarray:
        """
        Apply gradient-guided perturbation using dream-based mask for maximum effectiveness.
        
        Args:
            original_img: Original image (RGB, 0-255)
            dreamed_target: Dreamed target image (RGB, 0-255)
            protection_strength: Strength of perturbation (0.0-1.0)
            model: TensorFlow model for gradient computation
        
        Returns:
            Perturbed image
        """
        # Create mask from dream target patterns
        mask = self._create_dream_based_mask(original_img, dreamed_target)
        
        # Ensure dreamed target is the same size as original image
        if dreamed_target.shape != original_img.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        
        # Preprocess image for model
        img_for_model = self._load_image_from_array(original_img)
        
        # Compute gradients
        gradients = self._compute_gradients(img_for_model, model)
        
        # Convert gradients to numpy and normalize
        grad_np = gradients.numpy()
        # Remove batch dimension if present
        if grad_np.ndim == 4:
            grad_np = grad_np[0]  # Remove batch dimension
        grad_np = np.abs(grad_np)  # Use absolute values
        grad_np = grad_np / (np.max(grad_np) + 1e-8)  # Normalize
        
        # Resize gradients to match image size, channel-wise
        if grad_np.shape[:2] != (original_img.shape[0], original_img.shape[1]):
            grad_resized = np.stack([
                cv2.resize(grad_np[..., c], (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                for c in range(grad_np.shape[-1])
            ], axis=-1)
        else:
            grad_resized = grad_np
        
        # Apply masked gradient perturbation
        gradient_perturbation = grad_resized * protection_strength * 0.4 * mask
        
        # Apply masked dreamed target perturbation
        dreamed_perturbation = (dreamed_target.astype(np.float32) - original_img.astype(np.float32)) * protection_strength * 0.6 * mask
        
        # Combine perturbations
        perturbed = original_img.astype(np.float32) + dreamed_perturbation + gradient_perturbation
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        
        return perturbed

    def _load_image_from_array(self, img_array: np.ndarray) -> tf.Tensor:
        """
        Load image from numpy array for model input.
        
        Args:
            img_array: Image as numpy array (RGB, 0-255)
        
        Returns:
            Preprocessed tensor for model
        """
        # Resize to model input size
        img_resized = cv2.resize(img_array, (299, 299))
        
        # Preprocess for InceptionV3
        img_preprocessed = tf.keras.applications.inception_v3.preprocess_input(img_resized)
        
        # Add batch dimension
        img_tensor = tf.expand_dims(img_preprocessed, axis=0)
        
        return img_tensor

    def _create_structured_perlin_noise(self, width: int = 299, height: int = 299) -> tf.Tensor:
        """
        Create structured Perlin noise with wave patterns for variety.
        
        Args:
            width: Width of the noise image (default: 299)
            height: Height of the noise image (default: 299)
            
        Returns:
            Generated structured Perlin noise tensor of specified dimensions
        """
        # Create base Perlin noise
        base_noise = tf.random.normal((1, height, width, 3), mean=0.0, stddev=0.02)
        
        # Create multiple octaves with different characteristics
        noise_sum = base_noise[0]
        amplitude = 1.0
        
        for i in range(3):  # Fewer octaves for different character
            # Create noise at different frequencies
            freq_noise = tf.random.normal((1, height, width, 3), mean=0.0, stddev=0.02 * amplitude)
            # Apply frequency scaling
            freq_noise = tf.image.resize(freq_noise, (height, width))
            noise_sum += freq_noise[0] * amplitude
            
            amplitude *= 0.7  # Different decay rate
        
        # Add wave patterns for structure
        x = tf.range(width, dtype=tf.float32)
        y = tf.range(height, dtype=tf.float32)
        X, Y = tf.meshgrid(x, y)
        
        # Create different wave patterns
        wave1 = tf.sin(X * 0.15) * tf.cos(Y * 0.08) * 0.015
        wave2 = tf.sin(X * 0.06 + Y * 0.12) * 0.015
        wave3 = tf.cos(X * 0.03) * tf.sin(Y * 0.09) * 0.015
        
        # Add waves to each channel
        structured_noise = noise_sum + tf.stack([wave1, wave2, wave3], axis=-1)
        
        return structured_noise
    
    def _create_organic_lines_noise(self, width: int = 299, height: int = 299) -> np.ndarray:
        """
        Create organic lines noise pattern for linear noise type.
        
        Args:
            width: Width of the noise image
            height: Height of the noise image
            
        Returns:
            Generated organic lines noise array
        """
        # Create base noise
        noise = np.random.normal(0, 0.02, (height, width, 3))
        
        # Create organic line patterns using sine waves with varying frequencies
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Create multiple organic line patterns
        lines1 = np.sin(X * 0.1 + Y * 0.05) * np.cos(Y * 0.08) * 0.02
        lines2 = np.sin(X * 0.06 + Y * 0.12) * np.sin(Y * 0.04) * 0.015
        lines3 = np.cos(X * 0.08) * np.sin(Y * 0.1 + X * 0.03) * 0.018
        
        # Combine line patterns
        organic_lines = lines1 + lines2 + lines3
        
        # Add to each channel with slight variations
        organic_noise = noise + np.stack([organic_lines * 1.0, 
                                        organic_lines * 0.8, 
                                        organic_lines * 1.2], axis=-1)
        
        return organic_noise
    
    def _create_artistic_strokes_noise(self, width: int = 299, height: int = 299) -> np.ndarray:
        """
        Create artistic strokes noise pattern for linear noise type.
        
        Args:
            width: Width of the noise image
            height: Height of the noise image
            
        Returns:
            Generated artistic strokes noise array
        """
        # Create base noise
        noise = np.random.normal(0, 0.015, (height, width, 3))
        
        # Create artistic stroke patterns using more complex wave functions
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Create brush stroke patterns
        strokes1 = np.sin(X * 0.12) * np.cos(Y * 0.06 + X * 0.04) * 0.025
        strokes2 = np.cos(X * 0.05 + Y * 0.09) * np.sin(Y * 0.11) * 0.02
        strokes3 = np.sin(X * 0.08 + Y * 0.07) * np.cos(Y * 0.05) * 0.022
        
        # Create cross-hatching effect
        hatching1 = np.sin(X * 0.15) * np.sin(Y * 0.15) * 0.01
        hatching2 = np.cos(X * 0.1 + Y * 0.1) * 0.008
        
        # Combine stroke patterns
        artistic_strokes = strokes1 + strokes2 + strokes3 + hatching1 + hatching2
        
        # Add to each channel with artistic variations
        artistic_noise = noise + np.stack([artistic_strokes * 1.1, 
                                         artistic_strokes * 0.9, 
                                         artistic_strokes * 1.0], axis=-1)
        
        return artistic_noise
    
    def _generate_linear_noise_target(self, image_path: str) -> np.ndarray:
        """
        Generate combined organic lines and artistic strokes noise optimized for deep dream processing.
        Args:
            image_path: Path to the input image
        Returns:
            Generated combined organic-artistic noise image optimized for deep dream
        """
        # Load the image to get its dimensions
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image {image_path}. Supported formats: JPG, PNG, JPEG, BMP, TIFF")
        height, width, channels = img.shape
        # Generate dreamed target at the original image's resolution
        self._generate_dreamed_target(target_shape=(height, width, 3))
        return self.dreamed_target

    def poison(self, image_path, output_path, protection_strength=0.15, fgsm_epsilon=4/255, noise_type='perlin', frequency_band='high'):
        """
        Apply protection to an image using Perlin noise-dream perturbation with FGSM enhancement.
        Now supports gradient-guided perturbations for maximum model confusion.
        
        Args:
            image_path: Path to input image
            output_path: Path to save protected image
            protection_strength: Strength of protection (0.0-1.0)
            fgsm_epsilon: FGSM perturbation magnitude (default: 8/255, reduced from 16/255)
            noise_type: Type of noise to use ('perlin', 'linear', or 'fourier')
            frequency_band: Frequency band for Fourier perturbation ('low', 'mid', 'high', 'all')
        """
        # Load original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Failed to load image {image_path}. Supported formats: JPG, PNG, JPEG, BMP, TIFF")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Generate noise target based on type
        if noise_type == 'fourier':
            print(f"Generating optimized Fourier noise target...")
            dreamed_target = self._generate_fourier_noise_target(image_path)
        elif noise_type == 'linear':
            dreamed_target = self._generate_linear_noise_target(image_path)
        else:  # perlin (default)
            dreamed_target = self._generate_noise_target(image_path)
        
        # Apply gradient-guided perturbation for maximum model confusion
        print("  Applying gradient-guided perturbation...")
        perturbed_img = self._apply_masked_gradient_perturbation(
            original_img, dreamed_target, protection_strength, self.model
        )
        
        # Apply FGSM perturbation for additional robustness
        final_img = self._targeted_fgsm_perturbation(perturbed_img, dreamed_target, fgsm_epsilon)
        
        # Apply final Fourier perturbation pass for additional effectiveness
        print("  Applying final Fourier perturbation pass...")
        final_img = self._apply_fourier_perturbation(final_img, dreamed_target, protection_strength * 0.3, frequency_band)
        
        # Save the protected image
        self._save_image(final_img, output_path)
        
        return final_img
    
    def _apply_fourier_perturbation(self, original_img: np.ndarray, dreamed_target: np.ndarray, protection_strength: float, frequency_band: str = 'high') -> np.ndarray:
        """
        Apply perturbation in Fourier domain.
        
        Args:
            original_img: Original image (RGB, 0-255)
            dreamed_target: Dreamed target image (RGB, 0-255)
            protection_strength: Strength of perturbation (0.0-1.0)
            frequency_band: Which frequency band to target ('low', 'mid', 'high', 'all')
        
        Returns:
            Perturbed image
        """
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        
        height, width, channels = original_img.shape
        
        # Convert to float32 for FFT
        original_float = original_img.astype(np.float32) / 255.0
        dreamed_float = dreamed_target.astype(np.float32) / 255.0
        
        # Apply FFT to each channel
        perturbed_channels = []
        
        for c in range(channels):
            # FFT of original and dreamed channels
            orig_fft = fft2(original_float[:, :, c])
            dream_fft = fft2(dreamed_float[:, :, c])
            
            # Create frequency mask based on frequency_band
            freq_mask = self._create_frequency_mask(height, width, frequency_band)
            
            # Calculate perturbation in frequency domain
            fft_diff = dream_fft - orig_fft
            
            # Apply frequency mask and perturbation strength
            perturbation = fft_diff * freq_mask * protection_strength
            
            # Apply perturbation
            perturbed_fft = orig_fft + perturbation
            
            # Inverse FFT
            perturbed_channel = np.real(ifft2(perturbed_fft))
            
            # Clip to valid range
            perturbed_channel = np.clip(perturbed_channel, 0, 1)
            perturbed_channels.append(perturbed_channel)
        
        # Combine channels and convert back to uint8
        perturbed_img = np.stack(perturbed_channels, axis=-1)
        perturbed_img = (perturbed_img * 255).astype(np.uint8)
        
        return perturbed_img
    
    def _create_frequency_mask(self, height: int, width: int, frequency_band: str) -> np.ndarray:
        """
        Create frequency mask optimized for high-frequency protection.
        Prioritizes high-frequency components for maximum effectiveness.
        
        Args:
            height: Image height
            width: Image width
            frequency_band: Frequency band to target ('low', 'mid', 'high', 'all')
        
        Returns:
            Frequency mask array
        """
        # Create frequency coordinates
        u = np.fft.fftfreq(width)
        v = np.fft.fftfreq(height)
        U, V = np.meshgrid(u, v)
        
        # Calculate frequency magnitude
        freq_mag = np.sqrt(U**2 + V**2)
        max_freq = np.sqrt(0.5**2 + 0.5**2)
        freq_mag = freq_mag / max_freq
        
        # Create optimized frequency masks with high-frequency emphasis
        if frequency_band == 'low':
            # Low frequency mask (reduced emphasis)
            mask = np.where(freq_mag < 0.2, 1.0, 0.0)
        elif frequency_band == 'mid':
            # Mid frequency mask (moderate emphasis)
            mask = np.where((freq_mag >= 0.2) & (freq_mag < 0.4), 1.0, 0.0)
        elif frequency_band == 'high':
            # High frequency mask (strong emphasis - optimized for effectiveness)
            mask = np.where(freq_mag >= 0.3, 1.0, 0.0)
            # Add extra emphasis to very high frequencies
            very_high_mask = np.where(freq_mag >= 0.45, 1.5, 1.0)
            mask *= very_high_mask
        elif frequency_band == 'all':
            # All frequencies mask (with high-frequency emphasis)
            mask = np.ones_like(freq_mag)
            # Boost high frequencies
            high_boost = np.where(freq_mag >= 0.3, 1.3, 1.0)
            mask *= high_boost
        else:
            # Default to high frequency
            mask = np.where(freq_mag >= 0.3, 1.0, 0.0)
        
        # Apply smoothing to avoid artifacts
        mask = self._smooth_frequency_mask(mask, frequency_band)
        
        return mask
    
    def _smooth_frequency_mask(self, mask: np.ndarray, frequency_band: str) -> np.ndarray:
        """
        Apply smooth transition to frequency mask to avoid artifacts.
        
        Args:
            mask: Binary frequency mask
            frequency_band: Frequency band type
        
        Returns:
            Smoothed frequency mask
        """
        # Apply Gaussian smoothing to create smooth transitions
        from scipy.ndimage import gaussian_filter
        
        # Smooth the mask
        smoothed_mask = gaussian_filter(mask.astype(np.float32), sigma=2.0)
        
        # Normalize to [0, 1]
        if smoothed_mask.max() > 0:
            smoothed_mask = smoothed_mask / smoothed_mask.max()
        
        return smoothed_mask
    
    def _apply_adaptive_fourier_perturbation(self, original_img: np.ndarray, dreamed_target: np.ndarray, protection_strength: float) -> np.ndarray:
        """
        Apply adaptive Fourier perturbation that targets the most effective frequency bands.
        
        Args:
            original_img: Original image (RGB, 0-255)
            dreamed_target: Dreamed target image (RGB, 0-255)
            protection_strength: Strength of perturbation (0.0-1.0)
        
        Returns:
            Perturbed image
        """
        if original_img.shape != dreamed_target.shape:
            dreamed_target = cv2.resize(dreamed_target, (original_img.shape[1], original_img.shape[0]))
        
        height, width, channels = original_img.shape
        
        # Convert to float32 for FFT
        original_float = original_img.astype(np.float32) / 255.0
        dreamed_float = dreamed_target.astype(np.float32) / 255.0
        
        # Apply FFT to each channel
        perturbed_channels = []
        
        for c in range(channels):
            # FFT of original and dreamed channels
            orig_fft = fft2(original_float[:, :, c])
            dream_fft = fft2(dreamed_float[:, :, c])
            
            # Calculate frequency-dependent perturbation strength
            adaptive_strength = self._calculate_adaptive_strength(orig_fft, dream_fft, protection_strength)
            
            # Calculate perturbation in frequency domain
            fft_diff = dream_fft - orig_fft
            
            # Apply adaptive perturbation
            perturbation = fft_diff * adaptive_strength
            
            # Apply perturbation
            perturbed_fft = orig_fft + perturbation
            
            # Inverse FFT
            perturbed_channel = np.real(ifft2(perturbed_fft))
            
            # Clip to valid range
            perturbed_channel = np.clip(perturbed_channel, 0, 1)
            perturbed_channels.append(perturbed_channel)
        
        # Combine channels and convert back to uint8
        perturbed_img = np.stack(perturbed_channels, axis=-1)
        perturbed_img = (perturbed_img * 255).astype(np.uint8)
        
        return perturbed_img
    
    def _calculate_adaptive_strength(self, orig_fft: np.ndarray, dream_fft: np.ndarray, base_strength: float) -> np.ndarray:
        """
        Calculate adaptive perturbation strength optimized for high-frequency protection.
        Prioritizes high-frequency components for maximum effectiveness.
        
        Args:
            orig_fft: FFT of original image channel
            dream_fft: FFT of dreamed target channel
            base_strength: Base protection strength
        
        Returns:
            Adaptive strength mask optimized for high frequencies
        """
        height, width = orig_fft.shape
        
        # Create frequency coordinates
        u = np.fft.fftfreq(width)
        v = np.fft.fftfreq(height)
        U, V = np.meshgrid(u, v)
        
        # Calculate frequency magnitude
        freq_mag = np.sqrt(U**2 + V**2)
        max_freq = np.sqrt(0.5**2 + 0.5**2)
        freq_mag = freq_mag / max_freq
        
        # Calculate magnitude difference
        mag_diff = np.abs(dream_fft) - np.abs(orig_fft)
        
        # Create adaptive strength mask with high-frequency emphasis
        # Higher strength for frequencies with larger differences
        adaptive_mask = np.abs(mag_diff) / (np.abs(orig_fft) + 1e-8)
        adaptive_mask = np.clip(adaptive_mask, 0, 1)
        
        # Enhanced frequency weighting for high frequencies
        # Use exponential weighting to strongly favor high frequencies
        freq_weight = freq_mag ** 0.3  # Reduced exponent for stronger high-frequency emphasis
        
        # Add extra boost for very high frequencies
        very_high_boost = np.where(freq_mag >= 0.4, 1.5, 1.0)
        freq_weight *= very_high_boost
        
        # Combine factors with increased emphasis on high frequencies
        adaptive_strength = base_strength * adaptive_mask * freq_weight
        
        # Apply additional high-frequency boost
        high_freq_boost = np.where(freq_mag >= 0.35, 1.2, 1.0)
        adaptive_strength *= high_freq_boost
        
        return adaptive_strength
    
    def _create_fourier_noise(self, width: int = 299, height: int = 299) -> np.ndarray:
        """
        Create Fourier domain noise optimized for high-frequency deep dream processing.
        Focuses on strong high-frequency components for maximum effectiveness.
        
        Args:
            width: Width of the noise image
            height: Height of the noise image
            
        Returns:
            Generated high-frequency Fourier noise array
        """
        # Create base noise with higher intensity for high frequencies
        freq_noise = np.random.normal(0, 0.05, (height, width, 3))  # Increased from 0.02
        
        # Apply frequency-domain patterns optimized for high frequencies
        u = np.fft.fftfreq(width)
        v = np.fft.fftfreq(height)
        U, V = np.meshgrid(u, v)
        
        # Calculate frequency magnitude
        freq_mag = np.sqrt(U**2 + V**2)
        max_freq = np.sqrt(0.5**2 + 0.5**2)
        freq_mag = freq_mag / max_freq
        
        # Create high-frequency emphasis mask
        # Strong emphasis on high frequencies (freq_mag > 0.3)
        high_freq_mask = np.where(freq_mag > 0.3, 1.0, 0.1)
        
        # Create different high-frequency patterns for each channel
        for c in range(3):
            # High-frequency sine/cosine patterns with increased amplitude
            freq_pattern = np.sin(U * 20 + c * 3) * np.cos(V * 15 + c * 4) * 0.03  # Increased from 0.01
            freq_pattern += np.sin(U * 12 + V * 18) * 0.025  # Increased from 0.008
            
            # Add high-frequency noise patterns
            freq_pattern += np.sin(U * 30 + c * 5) * np.cos(V * 25 + c * 6) * 0.02
            freq_pattern += np.sin(U * 40 + V * 35) * 0.015
            
            # Apply high-frequency emphasis
            freq_pattern *= high_freq_mask
            
            # Add to noise
            freq_noise[:, :, c] += freq_pattern
        
        # Convert to spatial domain
        spatial_noise = np.zeros_like(freq_noise)
        for c in range(3):
            spatial_noise[:, :, c] = np.real(ifft2(freq_noise[:, :, c]))
        
        # Normalize and scale with higher intensity for high-frequency components
        spatial_noise = (spatial_noise - spatial_noise.min()) / (spatial_noise.max() - spatial_noise.min())
        spatial_noise = spatial_noise * 0.25  # Increased from 0.1 for stronger effect
        
        return spatial_noise
    
    def _generate_fourier_noise_target(self, image_path: str) -> np.ndarray:
        """
        Generate Fourier domain noise optimized for deep dream processing.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Generated Fourier noise image optimized for deep dream
        """
        # Load the image to get its dimensions
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image {image_path}. Supported formats: JPG, PNG, JPEG, BMP, TIFF")
        
        # Get image dimensions
        height, width, channels = img.shape
        
        # Create Fourier noise
        fourier_noise = self._create_fourier_noise(width, height)
        
        # Convert to tensor for deep dream processing
        noise_tensor = tf.convert_to_tensor(fourier_noise, dtype=tf.float32)
        
        # Run deep dream on the Fourier noise
        dreamed_fourier = self._run_deep_dream(
            img=noise_tensor,
            steps_per_octave=self.steps,
            step_size=self.step_size,
            octaves=self.num_ocataves,
            octave_scale=self.octave_scale
        )
        
        # Convert to RGB format
        dreamed_rgb = self._deprocess(dreamed_fourier.numpy())
        
        # Resize to match original image dimensions
        if dreamed_rgb.shape[:2] != (height, width):
            dreamed_rgb = cv2.resize(dreamed_rgb, (width, height))
        
        return dreamed_rgb