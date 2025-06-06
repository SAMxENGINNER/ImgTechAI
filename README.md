# ImgTechAI 

A powerful web-based image processing and computer vision application built with Streamlit. Transform your images using state-of-the-art AI and computer vision techniques through an intuitive, interactive interface.

## 🌟 Overview

ImgTechAI is a comprehensive image transformation tool that brings advanced computer vision capabilities to your browser. Whether you're a researcher, student, developer, or creative professional, this application provides easy access to a wide range of image processing techniques without requiring any coding knowledge.

## ✨ Key Features

### 🧠 Segmentation & Edge Detection
- **Binary Thresholding** - Simple threshold-based image segmentation
- **Adaptive Thresholding** - Locally adaptive thresholding for varying lighting
- **Otsu Thresholding** - Automatic optimal threshold selection
- **Watershed Segmentation** - Advanced technique for separating touching objects
- **K-Means Clustering** - Color-based image segmentation
- **Canny Edge Detection** - Industry-standard edge detection algorithm
- **Sobel Edge Detection** - Gradient-based edge detection

### 🌀 Fourier Transform & Frequency Analysis
- **2D Fast Fourier Transform (FFT)** - Convert images to frequency domain
- **Magnitude Spectrum** - Visualize frequency components
- **Phase Spectrum** - Display phase information
- **Frequency Filtering** - Low-pass, high-pass, and band-pass filters
- **Inverse FFT** - Reconstruct images from frequency domain

### 🔍 Filtering & Smoothing
- **Gaussian Blur** - Smooth noise reduction with configurable parameters
- **Image Sharpening** - Enhance image details and edges
- **Median Filter** - Effective salt-and-pepper noise removal

### 🎨 Color & Pixel Transformations
- **Grayscale Conversion** - Convert to monochrome images
- **Color Jitter** - Adjust brightness and contrast dynamically
- **Channel Shuffle** - Rearrange RGB color channels (6 different orders + random)
- **Hue Shift** - Modify color hue across the spectrum
- **Saturation Adjustment** - Control color intensity and vibrancy

### 🧪 Noise Injection
- **Gaussian Noise** - Add random noise with normal distribution
- **Salt & Pepper Noise** - Add random black and white pixels
- **Uniform Noise** - Add uniformly distributed random noise

### 🌈 Histogram & Contrast Enhancement
- **Histogram Equalization** - Improve overall image contrast
- **Gamma Correction** - Adjust brightness curves non-linearly
- **CLAHE** - Contrast Limited Adaptive Histogram Equalization
- **Logarithmic Transform** - Enhance details in dark regions

### 🔄 Geometric Transformations
- **Rotation** - Rotate images by any angle (-180° to +180°)
- **Scaling** - Resize images with intelligent cropping/padding
- **Shear Transform** - Apply shear deformation effects

### 📦 Advanced Image Augmentations
- **Cutout (Random Erasing)** - Remove random rectangular regions
- **Mixup Effect** - Blend images with random patterns

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd ImgTechAI
   ```

2. **Install required dependencies**
   ```bash
   pip install streamlit opencv-python numpy scikit-image pillow
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
ImgTechAI/
├── app.py                          # Main Streamlit application
├── static/
│   ├── IMAGE_TECH_AI.png          # Application logo
│   └── IMAGE_TECH_AI_favicon.png  # Browser favicon
├── README.md                       # This documentation
└── requirements.txt                # Python dependencies (optional)
```

## 🎯 How to Use

### Step 1: Upload Your Image
- Click the "📤 Upload an Image" button
- Select a JPG, JPEG, or PNG file from your device
- The image will be displayed in the main area

### Step 2: Choose Transformations
- Use the sidebar to select desired transformations
- Each category offers different techniques with customizable parameters
- Enable multiple transformations to create complex effects

### Step 3: Adjust Parameters
- Fine-tune transformation settings using interactive sliders
- See real-time previews as you adjust parameters
- Experiment with different combinations for unique results

### Step 4: Compare Results
- View original and transformed images side-by-side
- Check detailed image information (dimensions, channels, file size)
- Review the transformation summary at the bottom

### Step 5: Download Your Result
- Click the "📥 Download Transformed Image" button
- Save the processed image as a PNG file
- Use the transformed image in your projects

## 🔧 Technical Details

### Supported Image Formats
- **Input**: JPG, JPEG, PNG
- **Output**: PNG (high quality, lossless)

### Processing Pipeline
The application applies transformations in this optimized order:
1. Color & Pixel Transformations
2. Noise Injection
3. Histogram & Contrast Adjustments
4. Geometric Transformations
5. Filtering & Smoothing
6. Fourier Transform Operations
7. Image Augmentations
8. Segmentation & Edge Detection

### Performance Considerations
- Images are processed in-memory for speed
- Large images may take longer to process
- Complex transformations (like Fourier transforms) require more computation time

## 🎓 Educational Use Cases

### Computer Vision Learning
- Understand the effects of different filters and transformations
- Visualize frequency domain representations
- Learn about edge detection algorithms

### Image Preprocessing
- Prepare images for machine learning models
- Apply data augmentation techniques
- Normalize image characteristics

### Research & Development
- Prototype image processing pipelines
- Test algorithm parameters interactively
- Generate datasets with specific characteristics

### Creative Applications
- Create artistic effects and filters
- Experiment with color manipulations
- Generate unique visual content

## 🛠️ Dependencies

```python
streamlit      # Web application framework
opencv-python   # Computer vision library
numpy        # Numerical computing
scikit-image  # Image processing algorithms
pillow       # Image handling and I/O
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Adding New Features
- Implement additional image processing techniques
- Add new transformation categories
- Improve the user interface

### Bug Fixes
- Report issues through GitHub issues
- Submit pull requests with fixes
- Improve error handling

### Documentation
- Enhance code comments
- Update README with new features
- Create tutorials and examples

## 📊 Image Information Display

The application provides detailed information about both original and processed images:
- **Dimensions** - Width × Height in pixels
- **Color Channels** - Number of color channels (1 for grayscale, 3 for RGB)
- **Data Type** - Pixel value data type (typically uint8)
- **File Format** - Original format and output format
- **File Size** - Size in megabytes

## 🔍 Transformation Summary

After processing, view a complete summary of applied transformations including:
- Transformation names and categories
- Parameter values used
- Processing order
- Combined effects

## 📱 Browser Compatibility

ImgTechAI works best with modern web browsers:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🚨 Troubleshooting

### Common Issues

**Image won't upload**
- Check file format (must be JPG, JPEG, or PNG)
- Ensure file size is reasonable (< 50MB recommended)

**Slow processing**
- Large images take more time to process
- Complex transformations require more computation
- Try reducing image size or using fewer simultaneous transformations

**Application won't start**
- Verify all dependencies are installed
- Check Python version (3.7+ required)
- Ensure port 8501 is available

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) - Amazing web app framework
- Powered by [OpenCV](https://opencv.org/) - Computer vision library
- Enhanced with [scikit-image](https://scikit-image.org/) - Image processing toolkit
- Image handling by [Pillow](https://pillow.readthedocs.io/) - Python Imaging Library

---

**Ready to transform your images? Upload one now and start exploring! 🚀**
```
