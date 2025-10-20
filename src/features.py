import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

def geometric_features(img):
    """Compute basic geometric stats of binary signature."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    moments = cv2.moments(cnt)
    cx = moments['m10'] / moments['m00'] if moments['m00'] != 0 else 0
    cy = moments['m01'] / moments['m00'] if moments['m00'] != 0 else 0
    norm_cx, norm_cy = cx / w, cy / h
    return np.array([area, aspect_ratio, extent, norm_cx, norm_cy])

def hog_features(img):
    """HOG descriptor for shape and texture."""
    return hog(img, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

def lbp_features(img, P=8, R=1):
    """Local Binary Pattern histogram."""
    lbp = local_binary_pattern(img, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_features(img):
    """Combine geometric + HOG + LBP features."""
    geom = geometric_features(img)
    hogf = hog_features(img)
    lbpf = lbp_features(img)
    return np.concatenate([geom, hogf, lbpf])

