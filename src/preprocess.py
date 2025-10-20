import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def preprocess_image(img_path, out_size=(128, 256)):
    """
    Preprocess a signature image:
      1. Read grayscale
      2. Denoise
      3. Binarize
      4. Crop ROI
      5. Resize + pad
      6. Deskew (optional)
    Returns processed binary image
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {img_path}")

    # Step 1: Denoise slightly
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 2: Binarize (invert so ink = white)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Step 4: Crop signature region
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return cv2.resize(binary, out_size)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cropped = binary[y0:y1+1, x0:x1+1]

    # Step 5: Resize with aspect ratio preserved + padding
    h, w = cropped.shape
    scale = min(out_size[0] / h, out_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(cropped, (new_w, new_h))
    pad_h, pad_w = out_size[0] - new_h, out_size[1] - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return padded

def preprocess_dataset(input_dir, output_dir, out_size=(128, 256)):
    """
    Apply preprocessing to all images in a directory (recursively).
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                in_path = os.path.join(root, f)
                out_path = os.path.join(output_dir, f)
                processed = preprocess_image(in_path, out_size)
                cv2.imwrite(out_path, processed)



img_path = "D:\Signature_Project\src\original_2_1.png"
processed = preprocess_image(img_path)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), cmap='gray')

plt.subplot(1,2,2)
plt.title("Processed")
plt.imshow(processed, cmap='gray')
plt.show()
