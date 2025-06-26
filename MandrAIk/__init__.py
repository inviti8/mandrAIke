import os
from pathlib import Path
import click
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import random

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
        """Load and preprocess image."""
        img = Image.open(path).convert('RGB')
        img = img.resize((299, 299))
        img_array = np.array(img).astype(np.float32)
        return tf.keras.applications.inception_v3.preprocess_input(img_array)
    
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
    
    def _generate_dreamed_target(self):
        """Generate dreamed target from random noise if no target image provided."""
        print("No target image provided. Generating dreamed target from random noise...")
        
        # Create random noise as target
        random_target = tf.random.normal((1, 299, 299, 3), mean=0.0, stddev=0.1)
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
    
    def poison(self, image_path, target_image_path, output_path, protection_strength=0.15, fgsm_epsilon=16/255):
        """
        Apply protection to an image using target-dream perturbation with additional FGSM enhancement.
        
        Args:
            image_path: Path to input image
            target_image_path: Path to target image to dream
            output_path: Path to save protected image
            protection_strength: Strength of protection (0.0-1.0)
            fgsm_epsilon: FGSM perturbation magnitude (default: 16/255)
        """
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Dream the target image if not already done
        if self.dreamed_target is None:
            if target_image_path and os.path.exists(target_image_path):
                self.dreamed_target = self._dream_target_image(target_image_path)
            else:
                self._generate_dreamed_target()
        
        # Apply targeted perturbation based on dreamed target
        perturbed_img = self._apply_dreamed_perturbation(
            original_img, self.dreamed_target, protection_strength
        )
        
        # Apply additional FGSM perturbation for enhanced effectiveness
        # print(f"  Applying additional FGSM perturbation (epsilon: {fgsm_epsilon:.4f})...")
        # final_img = self._targeted_fgsm_perturbation(
        #     perturbed_img, self.dreamed_target, fgsm_epsilon
        # )
        
        # Save protected image
        self._save_image(final_img, output_path)
        
        return final_img
    
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
        Get dream output.
        
        Args:
            image_path: Path to input image
            output_path: Path to save protected image
        """
        # Load and preprocess image
        img = self._load_image(image_path)
        noised_img = self._noised_image(img)

        dreamed_img = self._run_deep_dream(
            img=noised_img, 
            steps_per_octave=self.steps,
            step_size=self.step_size,
            octaves=self.num_ocataves,
            octave_scale=self.octave_scale
        )

        # Convert to RGB and save
        dreamed_rgb = self._deprocess(dreamed_img.numpy())
        
        if output_path:
            self._save_image(dreamed_rgb, output_path)
        
        return dreamed_rgb
    
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
    
    def _apply_adversarial_noise(self, image, noise_level="Min"):
        """Apply adversarial noise to image."""
        noise_levels = {"Min": 0.05, "Medium": 0.1, "Max": 0.2}
        noise = np.random.normal(0, noise_levels[noise_level] * 255, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

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
        """Save image to path."""
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    def hallucinogen(self, image_path, target_image_path1, target_image_path2, output_path, protection_strength=0.15):
        """
        Apply enhanced chained hallucination protection with multi-space perturbations.
        
        Args:
            image_path: Path to input image
            target_image_path1: Path to first target image
            target_image_path2: Path to second target image
            output_path: Path to save protected image
            protection_strength: Strength of protection (0.0-1.0)
        """
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        print(f"Applying enhanced chained hallucination (multi-space)")
        
        # Initialize with original image
        current_img = original_img.copy()
        chain_steps = 20
        
        print(f"Applying enhanced chained hallucination (multi-space)")
        # Pre-dream both targets to avoid repeated computation
        print("  Pre-dreaming target images...")
        dreamed_target1 = self._dream_target_image(target_image_path1)
        
        # Dream target2 starting from dreamed_target1 for better coherence
        print("  Dreaming target2 from dreamed_target1 base...")
        dreamed_target2 = self._dream_target_image_from_base(target_image_path2, dreamed_target1)
        base_step_strength = protection_strength * 0.3
        for step in range(chain_steps):
            if step < 2:
                target_name = "Target 1"
                dreamed_target = dreamed_target1
                step_strength = base_step_strength * 1.2
            elif step < 4:
                target_name = "Target 2" if step % 2 == 0 else "Target 1"
                dreamed_target = dreamed_target2 if step % 2 == 0 else dreamed_target1
                step_strength = base_step_strength * 0.8
            else:
                target_name = "Target 2"
                dreamed_target = dreamed_target2
                step_strength = base_step_strength * 1.0
            print(f"  Chain step {step + 1}/{chain_steps}: Dreaming toward {target_name} (strength: {step_strength:.3f})")
            current_img = self._apply_dreamed_perturbation_lab(current_img, dreamed_target, step_strength)
            current_img = self._apply_dreamed_perturbation_hsv(current_img, dreamed_target, step_strength)

        current_img = self._apply_final_enhancement(current_img, protection_strength)

        # Save the final enhanced chained hallucination
        self._save_image(current_img, output_path)
        return current_img
    
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

    def fgsm_hallucinogen(self, image_path, target_image_path1, target_image_path2, output_path, epsilon=8/255, chain_steps=5):
        """
        Apply FGSM-style hallucination protection using dreamed targets.
        Args:
            image_path: Path to input image
            target_image_path1: Path to first target image
            target_image_path2: Path to second target image
            output_path: Path to save protected image
            epsilon: Step size for FGSM update (default 8/255)
            chain_steps: Number of chain steps (default 5)
        """
        import tensorflow as tf
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_shape = original_img.shape
        img = original_img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = tf.convert_to_tensor(img)

        # Pre-dream both targets
        dreamed_target1 = self._dream_target_image(target_image_path1)
        
        # Dream target2 starting from dreamed_target1 for better coherence
        dreamed_target2 = self._dream_target_image_from_base(target_image_path2, dreamed_target1)
        dreamed_target1 = cv2.resize(dreamed_target1, (img_shape[1], img_shape[0]))
        dreamed_target2 = cv2.resize(dreamed_target2, (img_shape[1], img_shape[0]))
        dreamed_target1 = dreamed_target1.astype(np.float32) / 255.0
        dreamed_target2 = dreamed_target2.astype(np.float32) / 255.0
        dreamed_target1 = tf.convert_to_tensor(np.expand_dims(dreamed_target1, axis=0))
        dreamed_target2 = tf.convert_to_tensor(np.expand_dims(dreamed_target2, axis=0))

        # FGSM chain
        perturbed = img
        for step in range(chain_steps):
            with tf.GradientTape() as tape:
                tape.watch(perturbed)
                # Alternate target per step
                if step % 2 == 0:
                    loss = tf.reduce_mean(tf.square(perturbed - dreamed_target1))
                else:
                    loss = tf.reduce_mean(tf.square(perturbed - dreamed_target2))
            grad = tape.gradient(loss, perturbed)
            signed_grad = tf.sign(grad)
            perturbed = perturbed + epsilon * signed_grad
            perturbed = tf.clip_by_value(perturbed, 0.0, 1.0)

        final_img = (perturbed[0].numpy() * 255).astype(np.uint8)
        self._save_image(final_img, output_path)
        return final_img

    def _dream_target_image_from_base(self, target_path: str, base_image: np.ndarray) -> np.ndarray:
        """Dream the target image starting from a base image for better coherence."""
        print(f"Dreaming target image from base: {target_path}")
        
        # Load target image
        target_img = self._load_image(target_path)
        
        # Resize base image to match target preprocessing
        base_resized = cv2.resize(base_image, (299, 299))
        base_tensor = tf.convert_to_tensor(base_resized.astype(np.float32))
        base_tensor = tf.keras.applications.inception_v3.preprocess_input(base_tensor)
        
        # Run deep dream starting from the base image
        dreamed_target = self._run_deep_dream(
            img=base_tensor,
            steps_per_octave=self.steps,
            step_size=self.step_size,
            octaves=self.num_ocataves,
            octave_scale=self.octave_scale
        )
        
        # Convert to RGB
        dreamed_rgb = self._deprocess(dreamed_target.numpy())
        return dreamed_rgb

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
        import tensorflow as tf
        
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
            # This creates conflicting spatial cues that confuse the model
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