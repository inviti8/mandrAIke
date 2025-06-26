# Import MandrAIk
from MandrAIk import MandrAIk
from pathlib import Path
import os

adversarial = MandrAIk()
FILE_PATH = Path(__file__).parent.parent
target_images = os.path.join(FILE_PATH, 'target_images')
test_results = os.path.join(FILE_PATH, 'test_results')

adversarial.dream(os.path.join(target_images, 'car_target_image1.jpg'), os.path.join(test_results, 'test_dream.jpg'))