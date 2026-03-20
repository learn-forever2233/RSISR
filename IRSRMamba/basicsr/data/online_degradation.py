import cv2
import numpy as np
import random

def random_blur(img, prob=0.8):
    """Random blur with different kernels"""
    if random.random() < prob:
        kernel_size = random.randint(3, 11) // 2 * 2 + 1
        sigma = random.uniform(0.2, 2.5)
        
        # Random kernel type
        kernel_type = random.choice(['gaussian', 'anisotropic', 'motion'])
        
        if kernel_type == 'gaussian':
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        elif kernel_type == 'anisotropic':
            # Anisotropic blur
            kernel = np.zeros((kernel_size, kernel_size))
            axis = random.choice([0, 1])
            kernel[kernel_size//2, :] = 1 / kernel_size if axis == 1 else 0
            kernel[:, kernel_size//2] = 1 / kernel_size if axis == 0 else 0
            img = cv2.filter2D(img, -1, kernel)
        elif kernel_type == 'motion':
            # Motion blur
            angle = random.uniform(0, 180)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1 / kernel_size
            # Rotate kernel
            M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            img = cv2.filter2D(img, -1, kernel)
    return img

def random_downsample(img, scale=4):
    """Random downsample with different interpolation methods"""
    # Scale jitter
    scale_jitter = random.uniform(3.8, 4.2)
    h, w = img.shape[:2]
    new_h, new_w = int(h / scale_jitter), int(w / scale_jitter)
    
    # Random interpolation
    interp = random.choices(['bicubic', 'bilinear', 'area'], weights=[0.5, 0.3, 0.2])[0]
    if interp == 'bicubic':
        img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif interp == 'bilinear':
        img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:  # area
        img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Resize back to original LQ size
    lq_h, lq_w = h // scale, w // scale
    img_down = cv2.resize(img_down, (lq_w, lq_h), interpolation=cv2.INTER_CUBIC)
    return img_down

def add_gaussian_noise(img, prob=0.5):
    """Add Gaussian noise"""
    if random.random() < prob:
        sigma = random.uniform(1, 10) / 255.
        noise = np.random.normal(0, sigma, img.shape)
        img = img + noise
        img = np.clip(img, 0, 1)
    return img

def add_poisson_noise(img, prob=0.2):
    """Add Poisson noise"""
    if random.random() < prob:
        scale = random.uniform(0.5, 2.0)
        # Poisson noise is dependent on image intensity
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        noise = np.random.poisson(img / 255. * scale) / scale * 255
        img = noise.astype(np.float32) / 255.
        img = np.clip(img, 0, 1)
    return img

def add_stripe_noise(img, prob=0.25):
    """Add stripe/banding noise"""
    if random.random() < prob:
        h, w = img.shape[:2]
        # Column bias
        col_bias = random.uniform(0.5, 4) / 255.
        col_pattern = np.sin(np.arange(w) * random.uniform(0.1, 0.5)) * col_bias
        col_pattern = np.tile(col_pattern[np.newaxis, :, np.newaxis], (h, 1, img.shape[2]))
        
        # Sinusoidal stripes
        amp = random.uniform(0.5, 3) / 255.
        stripe_pattern = np.sin(np.arange(h) * random.uniform(0.05, 0.2)) * amp
        stripe_pattern = np.tile(stripe_pattern[:, np.newaxis, np.newaxis], (1, w, img.shape[2]))
        
        img = img + col_pattern + stripe_pattern
        img = np.clip(img, 0, 1)
    return img

def add_fpn_nuc(img, prob=0.2):
    """Add FPN/NUC residual"""
    if random.random() < prob:
        h, w = img.shape[:2]
        # Row gain bias
        row_gain = 1 + random.uniform(-0.03, 0.03)
        row_pattern = np.linspace(1, row_gain, h)[:, np.newaxis, np.newaxis]
        
        # Column gain bias
        col_gain = 1 + random.uniform(-0.03, 0.03)
        col_pattern = np.linspace(1, col_gain, w)[np.newaxis, :, np.newaxis]
        
        img = img * row_pattern * col_pattern
        img = np.clip(img, 0, 1)
    return img

def add_hot_pixels(img, prob=0.1):
    """Add bad pixels/hot pixels"""
    if random.random() < prob:
        h, w = img.shape[:2]
        num_pixels = int(h * w * random.uniform(0.0005, 0.003))
        
        for _ in range(num_pixels):
            y = random.randint(0, h-1)
            x = random.randint(0, w-1)
            img[y, x] = random.uniform(0.8, 1.0)  # Bright hot pixels
    return img

def add_quantization(img, prob=0.3):
    """Add quantization/compression noise"""
    if random.random() < prob:
        # Random bit-depth
        bit_depth = random.randint(6, 8)
        max_val = 2**bit_depth - 1
        img = np.round(img * max_val) / max_val
    return img

def online_degradation(img_gt, scale=4):
    """Online degradation pipeline"""
    # Make a copy to avoid modifying the original
    img = img_gt.copy()
    
    # Step 1: Blur
    img = random_blur(img)
    
    # Step 2: Downsample
    img_lq = random_downsample(img, scale)
    
    # Step 3: Add noise
    img_lq = add_gaussian_noise(img_lq)
    img_lq = add_poisson_noise(img_lq)
    
    # Step 4: Add infrared-specific noise
    img_lq = add_stripe_noise(img_lq)
    img_lq = add_fpn_nuc(img_lq)
    img_lq = add_hot_pixels(img_lq)
    
    # Step 5: Quantization
    img_lq = add_quantization(img_lq)
    
    return img_lq