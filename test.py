import cv2
import numpy as np
import os
import random

# === PARAMETERS ===
image_dir = "dataset"
output_dir = "result"
image_size = (256, 256)

# Corrected SSIM(Structural Similarity Index)) function
def calculate_ssim(img1, img2, window_size=11, K1=0.01, K2=0.03, L=255):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = cv2.GaussianBlur(img1, (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(img2, (window_size, window_size), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (window_size, window_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (window_size, window_size), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

# Load images
def load_images(folder):

    image_files = [f for f in os.listdir(folder) if f.endswith(".png") or f.endswith(".jpg")]
    random.shuffle(image_files)
    images = []
    for filename in image_files:
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append((filename, img))
    return images

# Split data
def split_data(images, train_ratio=0.7):
    split_index = int(len(images) * train_ratio)
    return images[:split_index], images[split_index:]

# Manual KMeans
def manual_kmeans(img, k=3, max_iter=100):
    data = img.reshape((-1, 3)).astype(np.float32)
    centers = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(centers, new_centers, atol=1e-2):
            break
        centers = new_centers

    return labels.reshape(img.shape[:2]), centers

# Generate better shadow mask
def generate_shadow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5,5), 0)
    mask, _ = manual_kmeans(blurred, k=3)  

    cluster_means = [np.mean(hsv[mask == i][:,2]) for i in range(3)]
    shadow_cluster = int(np.argmin(cluster_means))
    shadow_mask = np.uint8(mask == shadow_cluster) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # smaller kernel
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_ERODE, kernel, iterations=1)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    return shadow_mask

def remove_shadow(img, mask):
    # Adaptive dilation depending on mask area
    mask_area = cv2.countNonZero(mask)
    h, w = mask.shape
    ratio = mask_area / (h * w)

    if ratio < 0.05:
        dilation_iter = 1
        inpaint_radius = 3
    elif ratio < 0.15:
        dilation_iter = 2
        inpaint_radius = 5
    else:
        dilation_iter = 3
        inpaint_radius = 7

    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=dilation_iter)
    shadow_area = cv2.inpaint(img, dilated, inpaint_radius, cv2.INPAINT_TELEA)
    return shadow_area

# Evaluate SSIM
def evaluate_ssim(gt_image, predicted_image):
    gt_gray = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2GRAY)
    return calculate_ssim(gt_gray, pred_gray)

# Process images
def process_images(images, phase="train"):
    scores = []
    for filename, img in images:
        shadow_mask = generate_shadow_mask(img)
        removed = remove_shadow(img, shadow_mask)
        
        if phase == "test":
            ssim_score = evaluate_ssim(img, removed)
            scores.append(ssim_score)
            print(f"{filename} - SSIM Score: {ssim_score:.4f}")

        result_img = np.hstack([
            cv2.resize(img, image_size),
            cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR),
            cv2.resize(removed, image_size)
        ])
        cv2.imwrite(os.path.join(output_dir, filename), result_img)

    return np.mean(scores) if scores else None

# === MAIN ===
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    images = load_images(image_dir)
    train_data, test_data = split_data(images)

    print("[INFO] Training Phase")
    process_images(train_data, phase="train")

    print("[INFO] Testing Phase")
    avg_ssim = process_images(test_data, phase="test")
    print(f"\n[FINAL AVERAGE SSIM on Test Set]: {avg_ssim:.4f}")
 
 