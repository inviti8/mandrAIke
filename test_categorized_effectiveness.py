#!/usr/bin/env python3
"""
Categorized Effectiveness Testing for MandrAIk

This test suite properly categorizes images and tests if MandrAIk prevents
correct recognition by AI systems.
"""

import unittest
import numpy as np
import tensorflow as tf
import cv2
import os
import tempfile
import shutil
from PIL import Image
import glob
from pathlib import Path
import sys

# Add the current directory to the path to import mandrAIk
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MandrAIk import *

FILE_PATH = Path(__file__).parent


class TestCategorizedEffectiveness(unittest.TestCase):
    """Test suite for categorized effectiveness evaluation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with categorized images."""
        cls.test_dir = tempfile.mkdtemp()
        cls.base_test_imgs = os.path.join(FILE_PATH, "test_images")
        cls.test_images_dir = os.path.join(cls.test_dir, "test_images")
        os.makedirs(cls.test_images_dir, exist_ok=True)
        
        # Copy test images
        if os.path.exists(cls.base_test_imgs):
            shutil.copytree(cls.base_test_imgs, cls.test_images_dir, dirs_exist_ok=True)
        
        # Initialize MandrAIk
        cls.mandraik = MandrAIk()
        
        # Load classifier
        cls.classifier = tf.keras.applications.InceptionV3(
            include_top=True, 
            weights='imagenet'
        )
        
        # Define test categories with expected classifications
        cls.test_categories = {
            "portraits": {
                "expected_classes": ["person", "face", "portrait", "human"],
                "keywords": ["portrait", "face", "person", "headshot", "selfie"],
                "images": []
            },
            "landscapes": {
                "expected_classes": ["mountain", "landscape", "nature", "forest", "beach", "sky"],
                "keywords": ["landscape", "mountain", "beach", "forest", "nature", "sky"],
                "images": []
            },
            "artwork": {
                "expected_classes": ["painting", "artwork", "drawing", "canvas", "art"],
                "keywords": ["painting", "artwork", "drawing", "canvas", "oil", "watercolor"],
                "images": []
            },
            "photography": {
                "expected_classes": ["camera", "photograph", "photo", "image"],
                "keywords": ["photo", "photography", "camera", "shot", "image"],
                "images": []
            },
            "objects": {
                "expected_classes": ["object", "thing", "item"],
                "keywords": ["object", "item", "thing", "tool", "device"],
                "images": []
            }
        }
        
        # Categorize available images
        cls.categorize_test_images()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def categorize_test_images(cls):
        """Categorize test images based on filename and content."""
        print("📁 Categorizing test images...")
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(cls.test_images_dir, ext)))
            all_images.extend(glob.glob(os.path.join(cls.test_images_dir, ext.upper())))
        
        if not all_images:
            print("⚠️  No test images found. Creating categorized sample images...")
            cls.create_categorized_sample_images()
            return
        
        # Categorize each image
        for img_path in all_images:
            filename = os.path.basename(img_path).lower()
            category_found = False
            
            for category, config in cls.test_categories.items():
                # Check if filename contains category keywords
                for keyword in config["keywords"]:
                    if keyword in filename:
                        config["images"].append(img_path)
                        category_found = True
                        print(f"  📁 {filename} → {category}")
                        break
                
                if category_found:
                    break
            
            # If no category found, try to classify by content
            if not category_found:
                predicted_category = cls.classify_image_by_content(img_path)
                if predicted_category:
                    cls.test_categories[predicted_category]["images"].append(img_path)
                    print(f"  🤖 {filename} → {predicted_category} (AI classified)")
                else:
                    cls.test_categories["objects"]["images"].append(img_path)
                    print(f"  ❓ {filename} → objects (uncategorized)")
        
        # Print summary
        print(f"\n📊 Image Categorization Summary:")
        for category, config in cls.test_categories.items():
            print(f"  {category}: {len(config['images'])} images")
    
    @classmethod
    def classify_image_by_content(cls, img_path):
        """Attempt to classify an image by its content using AI."""
        try:
            # Load and preprocess image
            img = cls.mandraik._load_image(img_path)
            predictions = cls.get_top_predictions(img, top_k=3)
            
            # Get top prediction
            top_class = predictions[0][1].lower()
            
            # Map prediction to category
            if any(word in top_class for word in ["person", "face", "human"]):
                return "portraits"
            elif any(word in top_class for word in ["mountain", "landscape", "nature", "forest", "beach", "sky"]):
                return "landscapes"
            elif any(word in top_class for word in ["painting", "artwork", "drawing", "canvas", "art"]):
                return "artwork"
            elif any(word in top_class for word in ["camera", "photograph", "photo"]):
                return "photography"
            else:
                return None
                
        except Exception as e:
            print(f"    Error classifying {img_path}: {e}")
            return None
    
    @classmethod
    def create_categorized_sample_images(cls):
        """Create categorized sample images for testing."""
        print("🎨 Creating categorized sample images...")
        
        # Create images for each category
        categories_to_create = {
            "portraits": cls.create_portrait_samples,
            "landscapes": cls.create_landscape_samples,
            "artwork": cls.create_artwork_samples,
            "photography": cls.create_photography_samples,
            "objects": cls.create_object_samples
        }
        
        for category, create_func in categories_to_create.items():
            for i in range(2):  # Create 2 images per category
                img_path = os.path.join(cls.test_images_dir, f"{category}_sample_{i+1}.jpg")
                create_func(img_path, i+1)
                cls.test_categories[category]["images"].append(img_path)
                print(f"  ✅ Created: {category}_sample_{i+1}.jpg")
    
    @classmethod
    def create_portrait_samples(cls, output_path, index):
        """Create sample portrait images."""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Background
        img[:, :] = [100, 120, 140]
        
        # Face shape
        cv2.ellipse(img, (256, 200), (80, 100), 0, 0, 360, (220, 200, 180), -1)
        
        # Eyes
        cv2.circle(img, (230, 180), 15, (255, 255, 255), -1)
        cv2.circle(img, (282, 180), 15, (255, 255, 255), -1)
        cv2.circle(img, (230, 180), 8, (0, 0, 0), -1)
        cv2.circle(img, (282, 180), 8, (0, 0, 0), -1)
        
        # Nose
        cv2.line(img, (256, 200), (256, 220), (200, 180, 160), 3)
        
        # Mouth
        cv2.ellipse(img, (256, 240), (20, 8), 0, 0, 180, (150, 100, 100), 2)
        
        # Add variation based on index
        if index == 2:
            # Add glasses
            cv2.ellipse(img, (230, 180), (25, 15), 0, 0, 360, (100, 100, 100), 3)
            cv2.ellipse(img, (282, 180), (25, 15), 0, 0, 360, (100, 100, 100), 3)
        
        Image.fromarray(img).save(output_path)
    
    @classmethod
    def create_landscape_samples(cls, output_path, index):
        """Create sample landscape images."""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        if index == 1:
            # Mountain landscape
            # Sky gradient
            for i in range(256):
                img[i, :] = [100 + i//2, 150 + i//2, 255 - i//3]
            
            # Mountains
            for i in range(256, 512):
                mountain_height = int(50 * np.sin(i/50) + 100)
                img[i, :] = [50 + mountain_height//2, 100 + mountain_height//3, 50 + mountain_height//4]
            
            # Sun
            cv2.circle(img, (400, 100), 30, (255, 255, 0), -1)
        else:
            # Beach landscape
            # Sky
            img[:200, :] = [135, 206, 235]
            
            # Ocean
            img[200:400, :] = [0, 105, 148]
            
            # Sand
            img[400:, :] = [238, 214, 175]
            
            # Sun
            cv2.circle(img, (400, 80), 25, (255, 255, 0), -1)
        
        Image.fromarray(img).save(output_path)
    
    @classmethod
    def create_artwork_samples(cls, output_path, index):
        """Create sample artwork images."""
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        if index == 1:
            # Abstract painting
            cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
            cv2.circle(img, (350, 150), 80, (0, 255, 0), -1)
            cv2.line(img, (100, 350), (400, 350), (0, 0, 255), 10)
        else:
            # Digital art
            for i in range(512):
                for j in range(512):
                    r = int(128 + 127 * np.sin(i/50) * np.cos(j/50))
                    g = int(128 + 127 * np.sin(i/30) * np.sin(j/30))
                    b = int(128 + 127 * np.cos(i/40) * np.cos(j/40))
                    img[i, j] = [r, g, b]
        
        Image.fromarray(img).save(output_path)
    
    @classmethod
    def create_photography_samples(cls, output_path, index):
        """Create sample photography images."""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        if index == 1:
            # Street photography style
            # Buildings
            cv2.rectangle(img, (100, 100), (200, 400), (100, 100, 100), -1)
            cv2.rectangle(img, (250, 150), (350, 400), (120, 120, 120), -1)
            cv2.rectangle(img, (400, 200), (450, 400), (80, 80, 80), -1)
            
            # Windows
            for x in [120, 140, 160, 180]:
                cv2.rectangle(img, (x, 120), (x+10, 140), (255, 255, 200), -1)
        else:
            # Wildlife photography style
            # Background
            img[:, :] = [50, 100, 50]
            
            # Animal silhouette
            cv2.ellipse(img, (256, 300), (80, 40), 0, 0, 360, (30, 30, 30), -1)
            cv2.circle(img, (200, 280), 20, (30, 30, 30), -1)
        
        Image.fromarray(img).save(output_path)
    
    @classmethod
    def create_object_samples(cls, output_path, index):
        """Create sample object images."""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        if index == 1:
            # Car-like object
            cv2.rectangle(img, (150, 300), (350, 400), (200, 0, 0), -1)
            cv2.circle(img, (200, 420), 30, (50, 50, 50), -1)
            cv2.circle(img, (300, 420), 30, (50, 50, 50), -1)
        else:
            # Furniture-like object
            cv2.rectangle(img, (200, 200), (300, 400), (139, 69, 19), -1)
            cv2.rectangle(img, (180, 180), (320, 200), (139, 69, 19), -1)
        
        Image.fromarray(img).save(output_path)
    
    def test_01_categorized_recognition_protection(self):
        """Test if MandrAIk prevents correct recognition by category."""
        print(f"\n🎯 Testing Categorized Recognition Protection")
        print(f"=" * 60)
        
        overall_results = {}
        
        for category, config in self.test_categories.items():
            if not config["images"]:
                print(f"⚠️  No images found for category: {category}")
                continue
            
            print(f"\n📁 Testing category: {category}")
            print(f"   Expected classes: {config['expected_classes']}")
            print(f"   Images to test: {len(config['images'])}")
            
            category_results = {
                'total_images': len(config['images']),
                'correctly_classified_original': 0,
                'correctly_classified_poisoned': 0,
                'protection_successes': 0,
                'confidence_drops': []
            }
            
            for img_path in config["images"]:
                img_name = os.path.basename(img_path)
                print(f"\n   Testing: {img_name}")
                
                # Get original predictions
                original_img = self.mandraik._load_image(img_path)
                original_predictions = self.get_top_predictions(original_img)
                original_top_class = original_predictions[0][1].lower()
                original_confidence = original_predictions[0][2]
                
                # Check if originally correctly classified
                was_correctly_classified = any(
                    expected in original_top_class 
                    for expected in config["expected_classes"]
                )
                
                if was_correctly_classified:
                    category_results['correctly_classified_original'] += 1
                    print(f"     Original: {original_top_class} ({original_confidence:.3f}) ✅ Correct")
                else:
                    print(f"     Original: {original_top_class} ({original_confidence:.3f}) ❌ Wrong")
                
                # Apply protection
                output_path = os.path.join(self.test_dir, f"poisoned_{img_name}")
                poisoned_img = self.mandraik.poison(img_path, output_path)
                
                # Get poisoned predictions
                poisoned_processed = self.mandraik._load_image(output_path)
                poisoned_predictions = self.get_top_predictions(poisoned_processed)
                poisoned_top_class = poisoned_predictions[0][1].lower()
                poisoned_confidence = poisoned_predictions[0][2]
                
                # Check if still correctly classified
                still_correctly_classified = any(
                    expected in poisoned_top_class 
                    for expected in config["expected_classes"]
                )
                
                if still_correctly_classified:
                    category_results['correctly_classified_poisoned'] += 1
                    print(f"     Poisoned: {poisoned_top_class} ({poisoned_confidence:.3f}) ❌ Still recognized")
                else:
                    print(f"     Poisoned: {poisoned_top_class} ({poisoned_confidence:.3f}) ✅ Protection worked!")
                
                # Calculate protection effectiveness
                if was_correctly_classified and not still_correctly_classified:
                    category_results['protection_successes'] += 1
                
                # Calculate confidence drop
                confidence_drop = original_confidence - poisoned_confidence
                category_results['confidence_drops'].append(confidence_drop)
                
                print(f"     Confidence drop: {confidence_drop:.3f}")
            
            # Calculate category statistics
            total_correct_original = category_results['correctly_classified_original']
            protection_successes = category_results['protection_successes']
            
            if total_correct_original > 0:
                protection_rate = protection_successes / total_correct_original
                avg_confidence_drop = np.mean(category_results['confidence_drops'])
                
                print(f"\n   📊 Category Results:")
                print(f"      Originally correct: {total_correct_original}/{category_results['total_images']}")
                print(f"      Protection successes: {protection_successes}/{total_correct_original}")
                print(f"      Protection rate: {protection_rate:.1%}")
                print(f"      Average confidence drop: {avg_confidence_drop:.3f}")
                
                # Assert minimum effectiveness
                self.assertGreater(protection_rate, 0.5, 
                    f"Protection rate for {category} should be >50%")
                self.assertGreater(avg_confidence_drop, 0.1,
                    f"Average confidence drop for {category} should be >10%")
            else:
                print(f"\n   ⚠️  No correctly classified images in original set")
            
            overall_results[category] = category_results
        
        # Calculate overall statistics
        self.print_overall_results(overall_results)
    
    def test_02_category_specific_effectiveness(self):
        """Test effectiveness for specific categories that matter to artists."""
        print(f"\n🎨 Testing Category-Specific Effectiveness")
        print(f"=" * 60)
        
        # Focus on categories most important to artists
        artist_categories = ["portraits", "artwork", "photography"]
        
        for category in artist_categories:
            if category not in self.test_categories or not self.test_categories[category]["images"]:
                continue
            
            print(f"\n🎯 Testing artist category: {category}")
            
            # Test with different protection strengths
            strengths = [0.1, 0.2, 0.3]
            
            for strength in strengths:
                print(f"   Protection strength: {strength}")
                
                success_count = 0
                total_tested = 0
                
                for img_path in self.test_categories[category]["images"][:3]:  # Test first 3
                    # Get original classification
                    original_img = self.mandraik._load_image(img_path)
                    original_predictions = self.get_top_predictions(original_img)
                    original_top_class = original_predictions[0][1].lower()
                    
                    # Check if originally correct
                    was_correct = any(
                        expected in original_top_class 
                        for expected in self.test_categories[category]["expected_classes"]
                    )
                    
                    if not was_correct:
                        continue
                    
                    total_tested += 1
                    
                    # Apply protection with specific strength
                    output_path = os.path.join(self.test_dir, f"strength_{strength}_{os.path.basename(img_path)}")
                    poisoned_img = self.mandraik.poison(img_path, output_path, strength)
                    
                    # Check if protection worked
                    poisoned_processed = self.mandraik._load_image(output_path)
                    poisoned_predictions = self.get_top_predictions(poisoned_processed)
                    poisoned_top_class = poisoned_predictions[0][1].lower()
                    
                    still_correct = any(
                        expected in poisoned_top_class 
                        for expected in self.test_categories[category]["expected_classes"]
                    )
                    
                    if not still_correct:
                        success_count += 1
                
                if total_tested > 0:
                    success_rate = success_count / total_tested
                    print(f"      Success rate: {success_rate:.1%} ({success_count}/{total_tested})")
                    
                    # Assert minimum effectiveness for artist categories
                    self.assertGreater(success_rate, 0.3, 
                        f"Protection rate for {category} at strength {strength} should be >30%")
    
    def print_overall_results(self, overall_results):
        """Print overall test results."""
        print(f"\n📊 OVERALL RESULTS")
        print(f"=" * 60)
        
        total_images = sum(r['total_images'] for r in overall_results.values())
        total_correct_original = sum(r['correctly_classified_original'] for r in overall_results.values())
        total_protection_successes = sum(r['protection_successes'] for r in overall_results.values())
        
        if total_correct_original > 0:
            overall_protection_rate = total_protection_successes / total_correct_original
            print(f"Total images tested: {total_images}")
            print(f"Originally correctly classified: {total_correct_original}")
            print(f"Protection successes: {total_protection_successes}")
            print(f"Overall protection rate: {overall_protection_rate:.1%}")
            
            # Calculate average confidence drop across all categories
            all_confidence_drops = []
            for results in overall_results.values():
                all_confidence_drops.extend(results['confidence_drops'])
            
            if all_confidence_drops:
                avg_confidence_drop = np.mean(all_confidence_drops)
                print(f"Average confidence drop: {avg_confidence_drop:.3f}")
            
            # Assert overall effectiveness
            self.assertGreater(overall_protection_rate, 0.4, 
                "Overall protection rate should be >40%")
        else:
            print("⚠️  No correctly classified images found in test set")
    
    def get_top_predictions(self, img, top_k=5):
        """Get top-k predictions for an image."""
        img_batch = tf.expand_dims(img, axis=0)
        predictions = self.classifier.predict(img_batch, verbose=0)
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.get_imagenet_class_name(idx)
            confidence = float(predictions[0][idx])
            results.append((idx, class_name, confidence))
        
        return results
    
    def get_imagenet_class_name(self, idx):
        """Get ImageNet class name for an index."""
        # Simplified mapping - in practice, load full ImageNet classes
        class_mapping = {
            0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
            4: "hammerhead_shark", 5: "electric_ray", 6: "stingray", 7: "rooster",
            8: "hen", 9: "ostrich", 10: "brambling", 11: "goldfinch", 12: "house_finch",
            13: "junco", 14: "indigo_bunting", 15: "American_robin", 16: "bulbul",
            17: "jay", 18: "magpie", 19: "chickadee", 20: "water_ouzel",
            # Add more mappings as needed
        }
        return class_mapping.get(idx, f"class_{idx}")


def run_categorized_evaluation():
    """Run categorized effectiveness evaluation."""
    print("🎯 MandrAIk Categorized Effectiveness Evaluation")
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    categorized_tests = loader.loadTestsFromTestCase(TestCategorizedEffectiveness)
    suite.addTests(categorized_tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    print("\n" + "=" * 60)
    print("CATEGORIZED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ Categorized evaluation completed successfully!")
    else:
        print("\n❌ Some tests failed.")
    
    return result


if __name__ == "__main__":
    run_categorized_evaluation() 