import os
from pathlib import Path
import click
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import random

class MandrAIk:
    def __init__(self, model_name='InceptionV3', steps=3, step_size=0.5, num_ocataves=3, octave_scale=2.5, noise_level="Min", layer_name='mixed1', max_dim=1024):
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
        self.steps = steps,
        self.step_size = step_size
        self.num_ocataves = num_ocataves
        self.octave_scale = octave_scale
        self.noise_level = noise_level
        self.layer_name = layer_name
        self.max_dim = max_dim
        self.model = self._load_model()
        self.noise_img = os.path.join(FILE_PATH, 'noised_image.jpg')
        
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

    
    def poison(self, image_path, output_path, protection_strength=0.15):
        """
        Apply protection to an image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save protected image
            protection_strength: Strength of protection (0.0-1.0)
        """
        # Load and preprocess image
        img = self._load_image(image_path)
        noised_img = self._apply_adversarial_noise(img)

        dreamed_img = self.run_deep_dream(
            img=noised_img, 
            model=self.model,
            steps_per_octave=self.steps,
            step_size=self.step_size,
            octaves=self.num_ocataves,
            octave_scale=self.octave_scale
        )
        # Convert images to RGB format for perturbation
        img_rgb = self._deprocess(img)
        dreamed_img_rgb = self._deprocess(dreamed_img.numpy())
        
        # Apply dream-based perturbation to the original image
        perturbed_img = self._apply_dream_based_perturbation(
            img_rgb, dreamed_img_rgb, 
            perturbation_strength=protection_strength
        )
        
        # Save protected image
        self._save_image(perturbed_img, output_path)
        
        return perturbed_img
    
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

        dreamed_img = self.run_deep_dream(
            img=noised_img, 
            model=self.model,
            steps_per_octave=self.steps,
            step_size=self.step_size,
            octaves=self.num_ocataves,
            octave_scale=self.octave_scale
        )

        dreamed_img = self._deprocess(dreamed_img.numpy())
        
        # Save protected image
        if output_path != None:
            self._save_image(dreamed_img, output_path)
        
        return dreamed_img
    
    def _noised_image(self, img):
        noised_img = self._apply_adversarial_noise(img)
        cv2.imwrite(self.noise_img, noised_img)
        return self._load_image(self.noise_img)
    
    def _apply_watermark(self, image, msg="ALL RIGHTS RESERVED"):
        binary_message = ''.join(format(ord(char), '08b') for char in msg)
        height, width, _ = image.shape
        total_pixels = height * width
        random.seed(42)
        pixel_indices = random.sample(range(total_pixels), len(binary_message))

        index = 0
        for pos in pixel_indices:
            x, y = divmod(pos, width)
            if index < len(binary_message):
                image[x, y, 2] = (image[x, y, 2] & 254) | int(binary_message[index])  # Modify Blue channel
                index += 1
        return image
    
    def _apply_adversarial_noise(self, image, noise_level="Min"):
        """
        Add adversarial noise with brightness correction.
        """
        noise_intensity = 1 if noise_level == "Min" else 5
        
        # Brightness correction - shift the range up
        image_brightened = image + 0.1  # Shift from [-1,1] to [-0.5,1.5]
        image_brightened = np.clip(image_brightened, -1, 1)  # Clip back to [-1,1]
        
        # Convert to 0-255 range
        image_255 = ((image_brightened + 1) * 127.5).astype(np.float32)
        
        # Fix BGR to RGB
        image_255 = image_255[:,:,[2,1,0]]
        
        # Generate noise
        noise = np.random.normal(0, noise_intensity, image.shape).astype(np.float32)
        
        # Add noise and ensure proper range
        result = image_255 + noise
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    def _apply_dream_based_perturbation(self, original_image, dreamed_image, perturbation_strength=0.1):
        """
        Apply color perturbation to the original image based on the dreamed image's RGB values.
        
        Args:
            original_image: numpy array of shape (height, width, 3) with values 0-255
            dreamed_image: numpy array of shape (height, width, 3) with values 0-255
            perturbation_strength: float between 0-1, controls how much the dream affects the original
        
        Returns:
            Perturbed image as numpy array
        """
        # Ensure both images are the same size
        if original_image.shape != dreamed_image.shape:
            dreamed_image = cv2.resize(dreamed_image, (original_image.shape[1], original_image.shape[0]))
        
        # Convert both to LAB color space
        original_lab = cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB)
        dreamed_lab = cv2.cvtColor(dreamed_image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        orig_l, orig_a, orig_b = cv2.split(original_lab)
        dream_l, dream_a, dream_b = cv2.split(dreamed_lab)
        
        # Calculate color differences (dream - original)
        a_diff = dream_a.astype(np.float32) - orig_a.astype(np.float32)
        b_diff = dream_b.astype(np.float32) - orig_b.astype(np.float32)
        
        # Apply perturbation based on dream differences
        new_a = np.clip(orig_a.astype(np.float32) + (a_diff * perturbation_strength), 0, 255).astype(np.uint8)
        new_b = np.clip(orig_b.astype(np.float32) + (b_diff * perturbation_strength), 0, 255).astype(np.uint8)
        
        # Keep original lightness (L channel) to maintain structure
        perturbed_lab = cv2.merge([orig_l, new_a, new_b])
        perturbed_rgb = cv2.cvtColor(perturbed_lab, cv2.COLOR_LAB2RGB)
        
        return perturbed_rgb

    def _calc_loss(self, img, model):
        img_batch = tf.expand_dims(img, axis=0)
        activation = model(img_batch)
        return tf.reduce_mean(activation)
    
    @tf.function
    def _deepdream_step(self, img, model, step_size):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self._calc_loss(img, model)
        gradients = tape.gradient(loss, img)
        gradients = gradients / (tf.math.reduce_std(gradients) + 1e-8)
        img = img + gradients * step_size
        img = tf.clip_by_value(img, -1, 1)
        return img, loss
    
    def run_deep_dream(self, img, model, steps_per_octave, step_size, octaves, octave_scale):
        base_shape = tf.shape(img)[:-1]
        img = tf.identity(img)
        for octave in range(octaves):
            new_size = tf.cast(tf.cast(base_shape, tf.float32) * (octave_scale ** octave), tf.int32)
            img = tf.image.resize(img, new_size)
            for step in steps_per_octave:
                img, loss = self._deepdream_step(img, model, step_size)
                if step % 10 == 0:
                    print(f"Octave {octave+1}/{octaves}, Step {step}, Loss: {loss.numpy():.4f}")
        return img
    
    
    def _deprocess(self, img):
        """Convert model output back to RGB."""
        img = img - img.min()
        img = img / img.max()
        return (img * 255).astype(np.uint8)
    
    def _load_image(self, path):
        """Load and preprocess image."""
        img = Image.open(path).convert('RGB')
        if self.max_dim:
            scale = self.max_dim / max(img.size)
            img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)))
        
        img_array = np.array(img).astype(np.float32)
        return tf.keras.applications.inception_v3.preprocess_input(img_array)
    
    def _save_image(self, img, path):
        """Save image to file."""
        Image.fromarray(img).save(path)