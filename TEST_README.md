# MandrAIk Effectiveness Test

This test suite evaluates the effectiveness of MandrAIk in creating adversarial images that prevent AI image recognition systems from correctly classifying protected images.

## Features

- **Modern Standards**: Uses state-of-the-art image classification models (InceptionV3, ResNet50, EfficientNet, Vision Transformer)
- **Comprehensive Metrics**: Measures attack success rate, confidence drop, accuracy, image quality, and processing time
- **Multiple Protection Strengths**: Tests low, medium, and high protection levels
- **Real-world Testing**: Works with real images from your `test_images` folder
- **Detailed Reporting**: Generates JSON results, Markdown reports, and visualizations

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r test_effectiveness_requirements.txt
   ```

2. **Prepare Test Images**:
   - Place your test images in the `test_images/` folder
   - Supported formats: JPG, JPEG, PNG, BMP, TIFF
   - If no images are found, the test will create sample images automatically

3. **Run the Test**:
   ```bash
   python test_mandrAIk_effectiveness.py
   ```

## Test Models

The test evaluates MandrAIk against these modern classification models:

### TensorFlow Models
- **InceptionV3**: Google's Inception architecture, widely used in production
- **ResNet50**: Deep residual network, excellent for general image classification
- **EfficientNetB0**: State-of-the-art efficiency-accuracy trade-off

### PyTorch Models
- **ResNet50**: PyTorch implementation with ImageNet weights
- **EfficientNet**: Modern efficient architecture
- **Vision Transformer (ViT)**: Latest transformer-based approach

## Metrics Explained

### Effectiveness Metrics
- **Attack Success Rate**: Percentage of images where the top predicted class changes after protection
- **Confidence Drop**: Average reduction in prediction confidence
- **Top-1 Accuracy**: Percentage of images where the original top prediction is maintained
- **Top-5 Accuracy**: Percentage of images where the original prediction remains in top-5

### Quality Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality degradation (higher is better)
- **SSIM (Structural Similarity Index)**: Measures perceptual similarity (closer to 1.0 is better)

### Performance Metrics
- **Processing Time**: Time taken to protect each image
- **Memory Usage**: Resource consumption during processing

## Output Files

After running the test, you'll find these files in the `test_results/` folder:

- `effectiveness_results.json`: Raw test data in JSON format
- `effectiveness_report.md`: Detailed human-readable report
- `effectiveness_analysis.png`: Visualization charts
- `protected_*.jpg`: Protected versions of your test images

## Customization

### Test Different Protection Strengths
```python
# In the test script, modify the protection_strengths parameter:
results = tester.run_comprehensive_test(['low', 'medium', 'high'])
```

### Add Your Own Models
```python
# Extend the ModernImageClassifier class to add new models
class CustomClassifier(ModernImageClassifier):
    def _load_model(self):
        # Load your custom model here
        pass
```

### Test Specific Image Categories
```python
# Organize your test_images folder by category:
test_images/
├── portraits/
├── landscapes/
├── artwork/
└── photography/
```

## Interpretation

### Good Results
- **High Attack Success Rate** (>80%): MandrAIk effectively prevents recognition
- **High Confidence Drop** (>0.3): Significant reduction in model confidence
- **Good PSNR** (>30dB): Minimal visual degradation
- **Good SSIM** (>0.8): High perceptual similarity

### Areas for Improvement
- **Low Attack Success Rate**: Consider increasing protection strength
- **Poor Image Quality**: Reduce protection strength or adjust parameters
- **Slow Processing**: Optimize MandrAIk parameters or use GPU acceleration

## Troubleshooting

### Common Issues
1. **CUDA/GPU Errors**: Install CPU-only versions of TensorFlow/PyTorch
2. **Memory Issues**: Reduce batch size or image dimensions
3. **Model Loading Errors**: Ensure internet connection for downloading weights

### Performance Tips
- Use GPU acceleration if available
- Reduce image dimensions for faster processing
- Test with fewer images initially

## Contributing

To improve the test suite:
1. Add new evaluation metrics
2. Include additional classification models
3. Implement new visualization types
4. Add support for different image formats

## License

This test suite is part of the MandrAIk project and follows the same license terms. 