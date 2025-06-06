import streamlit as st
import cv2
import numpy as np
from skimage import util
from PIL import Image
import io

st.set_page_config(page_title="ImgTechAI", layout="wide", page_icon="static/IMAGE_TECH_AI_favicon.png")
st.markdown('<p style="font-size:128px;">ImgTechAI</p>', unsafe_allow_html=True)

st.markdown("Transform your images with various AI and computer vision techniques!")

# Image
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.sidebar.image("static/IMAGE_TECH_AI.png", width=250)
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    transformed_img = img_np.copy()

    st.sidebar.title("🔧 Select Transformations")
    st.sidebar.markdown("---")

    # 🧠 Segmentation Preprocessing
    st.sidebar.subheader("🧠 Segmentation & Edge Detection")
    
    segmentation_method = st.sidebar.selectbox(
        "Choose segmentation technique:",
        ["None", "Binary Thresholding", "Adaptive Thresholding", "Otsu Thresholding",
        "Watershed", "K-Means", "Canny Edge Detection", "Sobel Edge Detection"]
    )
    # Parameters for segmentation methods
    if segmentation_method == "Binary Thresholding":
        binary_threshold = st.sidebar.slider("Binary Threshold", 0, 255, 127, 1)
    
    elif segmentation_method == "Adaptive Thresholding":
        adaptive_block_size = st.sidebar.slider("Block Size", 3, 21, 11, 2)
        adaptive_c = st.sidebar.slider("Constant C", -10, 10, 2, 1)
    
    elif segmentation_method == "Canny Edge Detection":
        canny_low = st.sidebar.slider("Canny Low Threshold", 0, 255, 100, 5)
        canny_high = st.sidebar.slider("Canny High Threshold", 0, 255, 200, 5)
    
    elif segmentation_method == "K-Means":
        k_clusters = st.sidebar.slider("Number of Clusters", 1, 50, 4, 1)
        k_iterations = st.sidebar.slider("Max Iterations", 1, 50, 10, 5)
    
    elif segmentation_method == "Watershed":
        watershed_threshold = st.sidebar.slider("Distance Transform Threshold", 0.1, 1.0, 0.7, 0.1)
    st.sidebar.markdown("---")

    #Fourier Tf & Frequency Analysis
    st.sidebar.subheader("🌀 Fourier Transform")
    
    apply_fourier = st.sidebar.checkbox("Apply Fourier Transform")
    if apply_fourier:
        fourier_display = st.sidebar.selectbox(
            "Display option:",
            ["Magnitude Spectrum", "Phase Spectrum", "Filtered Image"]
        )
        
        if fourier_display == "Filtered Image":
            filter_type = st.sidebar.selectbox(
                "Filter type:",
                ["Low Pass", "High Pass", "Band Pass"]
            )
            
            if filter_type == "Low Pass":
                cutoff_freq = st.sidebar.slider("Cutoff Frequency", 1, 100, 30, 1)
            elif filter_type == "High Pass":
                cutoff_freq = st.sidebar.slider("Cutoff Frequency", 1, 100, 30, 1)
            elif filter_type == "Band Pass":
                low_cutoff = st.sidebar.slider("Low Cutoff", 1, 50, 10, 1)
                high_cutoff = st.sidebar.slider("High Cutoff", 51, 100, 70, 1)
    
    st.sidebar.markdown("---")

    # 🔍 Filtering & Smoothing
    st.sidebar.subheader("🔍 Filtering & Smoothing")
    
    apply_blur = st.sidebar.checkbox("Gaussian Blur")
    if apply_blur:
        blur_kernel = st.sidebar.slider("Blur Kernel Size", 1, 31, 15, 2)
        blur_sigma = st.sidebar.slider("Blur Sigma", 0.1, 10.0, 2.0, 0.1)
    
    apply_sharpen = st.sidebar.checkbox("Sharpen")
    if apply_sharpen:
        sharpen_strength = st.sidebar.slider("Sharpen Strength", 0.1, 3.0, 1.0, 0.1)
    
    apply_median_filter = st.sidebar.checkbox("Median Filter")
    if apply_median_filter:
        median_kernel = st.sidebar.slider("Median Kernel Size", 3, 15, 5, 2)

    st.sidebar.markdown("---")

    #Color & Pixel Transformations
    st.sidebar.subheader("🎨 Color & Pixel Transformations")
    
    apply_grayscale = st.sidebar.checkbox("Grayscale")
    
    apply_color_jitter = st.sidebar.checkbox("Color Jitter")
    if apply_color_jitter:
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.2, 0.1)
        contrast = st.sidebar.slider("Contrast Offset", -50, 50, 30, 5)
    
    apply_hue_shift = st.sidebar.checkbox("Hue Shift")
    if apply_hue_shift:
        hue_shift = st.sidebar.slider("Hue Shift Amount", -180, 180, 30, 10)
    
    apply_saturation = st.sidebar.checkbox("Saturation Adjustment")
    if apply_saturation:
        saturation_factor = st.sidebar.slider("Saturation Factor", 0.0, 3.0, 1.5, 0.1)

    apply_channel_shuffle = st.sidebar.checkbox("Channel Shuffle")
    if apply_channel_shuffle:
        channel_order = st.sidebar.selectbox(
            "Channel Order",
            ["RGB (Original)", "RBG", "GRB", "GBR", "BRG", "BGR", "Random"]
        )

    st.sidebar.markdown("---")

    # Noise Injection
    st.sidebar.subheader("🧪 Noise Injection")
    
    add_gaussian_noise = st.sidebar.checkbox("Add Gaussian Noise")
    if add_gaussian_noise:
        noise_std = st.sidebar.slider("Noise Standard Deviation", 1, 100, 25, 1)
    
    add_sp_noise = st.sidebar.checkbox("Add Salt & Pepper Noise")
    if add_sp_noise:
        sp_amount = st.sidebar.slider("Salt & Pepper Amount", 0.01, 0.3, 0.05, 0.01)
    
    add_uniform_noise = st.sidebar.checkbox("Add Uniform Noise")
    if add_uniform_noise:
        uniform_low = st.sidebar.slider("Uniform Noise Low", -50, 0, -25, 5)
        uniform_high = st.sidebar.slider("Uniform Noise High", 0, 50, 25, 5)

    st.sidebar.markdown("---")

    #Histogram & Contrast
    st.sidebar.subheader("🌈 Histogram & Contrast")
    
    apply_hist_eq = st.sidebar.checkbox("Histogram Equalization")
    
    apply_gamma = st.sidebar.checkbox("Gamma Correction")
    if apply_gamma:
        gamma_value = st.sidebar.slider("Gamma Value", 0.1, 3.0, 1.5, 0.1)
    
    apply_clahe = st.sidebar.checkbox("CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    if apply_clahe:
        clip_limit = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0, 0.5)
        tile_grid_size = st.sidebar.slider("Tile Grid Size", 4, 16, 8, 2)
    
    apply_log_transform = st.sidebar.checkbox("Log Transform")
    if apply_log_transform:
        log_constant = st.sidebar.slider("Log Constant", 1, 100, 50, 1)

    st.sidebar.markdown("---")

    #Geometric Transformations
    st.sidebar.subheader("🔄 Geometric Transformations")
    
    apply_rotation = st.sidebar.checkbox("Rotation")
    if apply_rotation:
        rotation_angle = st.sidebar.slider("Rotation Angle (degrees)", -180, 180, 45, 5)
    
    apply_scaling = st.sidebar.checkbox("Scaling")
    if apply_scaling:
        scale_factor = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.5, 0.1)
    
    apply_shear = st.sidebar.checkbox("Shear Transform")
    if apply_shear:
        shear_factor = st.sidebar.slider("Shear Factor", -1.0, 1.0, 0.3, 0.1)

    st.sidebar.markdown("---")


    #Augmentation
    st.sidebar.subheader("📦 Augmentations")
    
    apply_cutout = st.sidebar.checkbox("Cutout (Random Erasing)")
    if apply_cutout:
        cutout_size = st.sidebar.slider("Cutout Size", 10, 200, 50, 10)
        num_cutouts = st.sidebar.slider("Number of Cutouts", 1, 10, 1, 1)
    
    apply_mixup = st.sidebar.checkbox("Mixup Effect")
    if apply_mixup:
        mixup_alpha = st.sidebar.slider("Mixup Alpha", 0.1, 1.0, 0.5, 0.1)

    st.sidebar.markdown("---")

    # === Apply Transformations
    
    # Color & Pixel Transformations
    if apply_grayscale:
        transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2GRAY)
        transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_GRAY2RGB)

    if apply_color_jitter:
        transformed_img = cv2.convertScaleAbs(transformed_img, alpha=brightness, beta=contrast)

    if apply_channel_shuffle== True:
        if channel_order == "RGB (Original)":
            pass
        elif channel_order == "RBG":
            transformed_img = transformed_img[..., [0, 2, 1]]
        elif channel_order == "GRB":
            transformed_img = transformed_img[..., [1, 0, 2]]
        elif channel_order == "GBR":
            transformed_img = transformed_img[..., [1, 2, 0]]
        elif channel_order == "BRG":
            transformed_img = transformed_img[..., [2, 0, 1]]
        elif channel_order == "BGR":
            transformed_img = transformed_img[..., [2, 1, 0]]
        else:  # Random
            transformed_img = transformed_img[..., np.random.permutation(3)]

    if apply_hue_shift:
        hsv = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        transformed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    if apply_saturation:
        hsv = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        transformed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Noise Injection
    if add_gaussian_noise:
        noise = np.random.normal(0, noise_std, transformed_img.shape).astype(np.int16)
        transformed_img = np.clip(transformed_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if add_sp_noise:
        transformed_img = util.random_noise(transformed_img, mode='s&p', amount=sp_amount)
        transformed_img = (255 * transformed_img).astype(np.uint8)
    
    if add_uniform_noise:
        noise = np.random.uniform(uniform_low, uniform_high, transformed_img.shape).astype(np.int16)
        transformed_img = np.clip(transformed_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Histogram & Contrast
    if apply_hist_eq:
        yuv = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        transformed_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    if apply_gamma:
        invGamma = 1.0 / gamma_value
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        transformed_img = cv2.LUT(transformed_img, table)

    if apply_clahe:
        lab = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        transformed_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    if apply_log_transform:
        c = log_constant
        transformed_img = c * np.log(1 + transformed_img.astype(np.float32))
        transformed_img = np.clip(transformed_img, 0, 255).astype(np.uint8)

    # Geometric Transformations
    if apply_rotation:
        h, w = transformed_img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        transformed_img = cv2.warpAffine(transformed_img, rotation_matrix, (w, h))
    
    if apply_scaling:
        h, w = transformed_img.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        transformed_img = cv2.resize(transformed_img, (new_w, new_h))
        # Crop or pad to original size
        if scale_factor > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            transformed_img = transformed_img[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            transformed_img = cv2.copyMakeBorder(transformed_img, pad_h, h-new_h-pad_h, 
                                               pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT)
    
    if apply_shear:
        h, w = transformed_img.shape[:2]
        shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        transformed_img = cv2.warpAffine(transformed_img, shear_matrix, (w, h))

    # Filtering & Smoothing
    if apply_blur:
        if blur_kernel % 2 == 0:
            blur_kernel += 1  # Ensure odd kernel size
        transformed_img = cv2.GaussianBlur(transformed_img, (blur_kernel, blur_kernel), blur_sigma)
    
    if apply_sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * sharpen_strength
        kernel[1, 1] = kernel[1, 1] - sharpen_strength + 1
        transformed_img = cv2.filter2D(transformed_img, -1, kernel)
        transformed_img = np.clip(transformed_img, 0, 255).astype(np.uint8)
    
    if apply_median_filter:
        if median_kernel % 2 == 0:
            median_kernel += 1  # Ensure odd kernel size
        transformed_img = cv2.medianBlur(transformed_img, median_kernel)

    # Fourier Transform
    if apply_fourier:
        # Convert to grayscale for Fourier Transform
        if len(transformed_img.shape) == 3:
            gray_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = transformed_img
            
        # Apply 2D FFT
        f_transform = np.fft.fft2(gray_img)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude and phase spectrum
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        phase_spectrum = np.angle(f_shift)
        
        # Normalize for display
        magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        phase_spectrum_normalized = cv2.normalize(phase_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create filter mask
        rows, cols = gray_img.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        
        if fourier_display == "Filtered Image":
            if filter_type == "Low Pass":
                mask[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 1
                mask = 1 - mask
            elif filter_type == "High Pass":
                mask[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 0
            elif filter_type == "Band Pass":
                mask[crow-high_cutoff:crow+high_cutoff, ccol-high_cutoff:ccol+high_cutoff] = 1
                mask[crow-low_cutoff:crow+low_cutoff, ccol-low_cutoff:ccol+low_cutoff] = 0
            
            # Apply filter
            f_shift_filtered = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            # Normalize for display
            filtered_img = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            transformed_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
        else:
            # Display spectrum
            if fourier_display == "Magnitude Spectrum":
                spectrum_img = magnitude_spectrum_normalized
            else:  # Phase Spectrum
                spectrum_img = phase_spectrum_normalized
                
            transformed_img = cv2.cvtColor(spectrum_img, cv2.COLOR_GRAY2RGB)

    # Augmentations
    if apply_cutout:
        h, w, _ = transformed_img.shape
        for _ in range(num_cutouts):
            if h > cutout_size and w > cutout_size:
                x = np.random.randint(0, w - cutout_size)
                y = np.random.randint(0, h - cutout_size)
                transformed_img[y:y+cutout_size, x:x+cutout_size] = 0
    
    if apply_mixup:
        # Create a simple pattern for mixup effect
        h, w, c = transformed_img.shape
        pattern = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
        transformed_img = (mixup_alpha * transformed_img + (1 - mixup_alpha) * pattern).astype(np.uint8)

    # Segmentation and Edge Detection
    if segmentation_method != "None":
        gray = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2GRAY)

        if segmentation_method == "Binary Thresholding":
            _, thresh = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
            transformed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        elif segmentation_method == "Adaptive Thresholding":
            if adaptive_block_size % 2 == 0:
                adaptive_block_size += 1  # Ensure odd block size
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
            transformed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        elif segmentation_method == "Otsu Thresholding":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            transformed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        elif segmentation_method == "Canny Edge Detection":
            edges = cv2.Canny(gray, canny_low, canny_high)
            transformed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        elif segmentation_method == "Sobel Edge Detection":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = np.clip(sobel, 0, 255).astype(np.uint8)
            transformed_img = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

        elif segmentation_method == "K-Means":
            Z = transformed_img.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, k_iterations, 1.0)
            _, label, center = cv2.kmeans(Z, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            transformed_img = res.reshape((transformed_img.shape))

        elif segmentation_method == "Watershed":
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, watershed_threshold * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            markers = cv2.connectedComponents(sure_fg)[1]
            markers = markers + 1
            markers[unknown == 255] = 0
            transformed_img_bgr = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
            markers = cv2.watershed(transformed_img_bgr, markers)
            transformed_img[markers == -1] = [255, 0, 0]  # red boundary

    # === Display Images ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(img_np, use_container_width=True)
        size_in_mb = uploaded_file.size / (1024 * 1024)
        # Display original image info
        st.info(f"**Original Image Info:**\n"
                f"- Size: {img_np.shape[1]} × {img_np.shape[0]} pixels\n"
                f"- Channels: {img_np.shape[2]}\n"
                f"- Data type: {img_np.dtype}\n"
                f"- Format: {uploaded_file.type}\n"
                f"- Size: {size_in_mb:.2f} MB\n")
    
    with col2:
        st.subheader("🛠️ Transformed Image")
        st.image(transformed_img, use_container_width=True)
        
        # Display transformed image info
        if len(transformed_img.shape) == 3:
            channels = transformed_img.shape[2]
        else:
            channels = 1
        pil_img = Image.fromarray(transformed_img)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf_size = buf.tell()  
        size_in_mb = buf_size / (1024 * 1024)
        st.info(f"**Transformed Image Info:**\n"
                f"- Size: {transformed_img.shape[1]} × {transformed_img.shape[0]} pixels\n"
                f"- Channels: {channels}\n"
                f"- Data type: {transformed_img.dtype}\n"
                f"- Format: image/png\n"
                f"- Size: {size_in_mb:.2f} MB\n")

    # === Download Transformed Image ===
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        result_img = Image.fromarray(transformed_img)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="📥 Download Transformed Image",
            data=byte_im,
            file_name="transformed_image.png",
            mime="image/png",
            use_container_width=True
        )

    # === Transformation Summary ===
    st.markdown("---")
    st.subheader("📊 Applied Transformations Summary")
    
    applied_transforms = []
    if apply_grayscale: applied_transforms.append("Grayscale")
    if apply_color_jitter: applied_transforms.append(f"Color Jitter (Brightness: {brightness}, Contrast: {contrast})")
    if apply_channel_shuffle: applied_transforms.append("Channel Shuffle")
    if apply_hue_shift: applied_transforms.append(f"Hue Shift ({hue_shift}°)")
    if apply_saturation: applied_transforms.append(f"Saturation Adjustment ({saturation_factor}x)")
    if add_gaussian_noise: applied_transforms.append(f"Gaussian Noise (σ={noise_std})")
    if add_sp_noise: applied_transforms.append(f"Salt & Pepper Noise ({sp_amount})")
    if add_uniform_noise: applied_transforms.append(f"Uniform Noise ({uniform_low} to {uniform_high})")
    if apply_hist_eq: applied_transforms.append("Histogram Equalization")
    if apply_gamma: applied_transforms.append(f"Gamma Correction (γ={gamma_value})")
    if apply_clahe: applied_transforms.append(f"CLAHE (Clip: {clip_limit}, Grid: {tile_grid_size})")
    if apply_log_transform: applied_transforms.append(f"Log Transform (c={log_constant})")
    if apply_rotation: applied_transforms.append(f"Rotation ({rotation_angle}°)")
    if apply_scaling: applied_transforms.append(f"Scaling ({scale_factor}x)")
    if apply_shear: applied_transforms.append(f"Shear Transform ({shear_factor})")
    if apply_blur: applied_transforms.append(f"Gaussian Blur (Kernel: {blur_kernel}, σ={blur_sigma})")
    if apply_sharpen: applied_transforms.append(f"Sharpen (Strength: {sharpen_strength})")
    if apply_median_filter: applied_transforms.append(f"Median Filter (Kernel: {median_kernel})")
    if apply_cutout: applied_transforms.append(f"Cutout ({num_cutouts} cuts of size {cutout_size})")
    if apply_mixup: applied_transforms.append(f"Mixup Effect (α={mixup_alpha})")
    if segmentation_method != "None": applied_transforms.append(f"Segmentation: {segmentation_method}")
    if apply_fourier:
        if fourier_display == "Magnitude Spectrum":
            applied_transforms.append("Fourier Transform: Magnitude Spectrum")
        elif fourier_display == "Phase Spectrum":
            applied_transforms.append("Fourier Transform: Phase Spectrum")
        else:
            if filter_type == "Low Pass":
                applied_transforms.append(f"Fourier Transform: Low Pass Filter (Cutoff: {cutoff_freq})")
            elif filter_type == "High Pass":
                applied_transforms.append(f"Fourier Transform: High Pass Filter (Cutoff: {cutoff_freq})")
            elif filter_type == "Band Pass":
                applied_transforms.append(f"Fourier Transform: Band Pass Filter (Low: {low_cutoff}, High: {high_cutoff})")
    
    if applied_transforms:
        for i, transform in enumerate(applied_transforms, 1):
            st.write(f"{i}. {transform}")
    else:
        st.write("No transformations applied.")

else:
    st.info("👆 Please upload an image to start transforming!")
    
    # Show example transformations
    st.markdown("---")
    st.subheader("🎯 Available Transformations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        
        ### 🧠 Segmentation & Edge Detection
        - **Thresholding** (Binary, Adaptive, Otsu)
        - **Watershed Segmentation**
        - **K-Means Clustering**
        - **Edge Detection** (Canny, Sobel)
                    
        ### 🌀 Fourier Transform & Frequency Analysis
        - **2D FFT** (Fast Fourier Transform)
        - **Magnitude Spectrum Visualization**
        - **Inverse FFT (Image Reconstruction)**
        - **Frequency Domain Filtering**
                    
        ### 🔍 Filtering & Smoothing
        - **Gaussian Blur**
        - **Sharpening**
        - **Median Filter**
                
        ### 🎨 Color & Pixel Transformations
        - **Grayscale Conversion**
        - **Color Jitter** (Brightness / Contrast)
        - **Channel Shuffle**
        - **Hue Shift**
        - **Saturation Adjustment**

        
        """)

    with col2:
        st.markdown("""
        ### 🧪 Noise Injection
        - **Gaussian Noise**
        - **Salt & Pepper Noise**
        - **Uniform Noise**

        ### 🌈 Histogram & Contrast Enhancements
        - **Histogram Equalization**
        - **Gamma Correction**
        - **CLAHE (Adaptive Histogram Equalization)**
        - **Logarithmic Transform**
                    
        ### 📐 Geometric Transformations
        - **Rotation**
        - **Scaling**
        - **Shear Transform**


        ### 📦 Image Augmentations
        - **Cutout / Random Erasing**
        - **Mixup Effect**


        """)
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Upload an image and experiment with various transformation parameters!</p>
    </div>
    """, 
    unsafe_allow_html=True
)
