import cv2
import numpy as np
import os

# ------------- CONFIG ------------- #
INPUT_FOLDER = 'dataset'           # folder with your collected shadow images
OUTPUT_FOLDER = 'images_preprocessed' # folder to save preprocessed images
IMG_SIZE = (256, 256)                 # resize dimension
APPLY_AUGMENTATION = True             # optional random flip
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------- FUNCTIONS ------------- #

def preprocess_image(img_path, apply_aug=False):
    """Resize, normalize, grayscale, and optional flip."""
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_gray = cv2.cvtColor((img_normalized * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if apply_aug and np.random.rand() > 0.5:
        img_resized = cv2.flip(img_resized, 1)
        img_gray = cv2.flip(img_gray, 1)

    return img_resized, img_gray

def cross_correlation_manual(img1, img2):
    """Compute normalized cross-correlation between two grayscale images manually."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 = (img1 - np.mean(img1)) / (np.std(img1) + 1e-5)
    img2 = (img2 - np.mean(img2)) / (np.std(img2) + 1e-5)
    corr = np.mean(img1 * img2)
    return corr


def create_heatmap(matrix, save_path):
    """Create and save a simple heatmap using OpenCV."""
    norm = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)

def draw_boxplot(data, save_path):
    """Simple boxplot drawing (vertical line for min/25%/median/75%/max) using OpenCV."""
    canvas = np.ones((400, 600, 3), dtype=np.uint8) * 255

    data_sorted = np.sort(data)
    n = len(data_sorted)
    min_val = data_sorted[0]
    max_val = data_sorted[-1]
    q1 = data_sorted[n//4]
    median = data_sorted[n//2]
    q3 = data_sorted[(3*n)//4]

    def map_val(val, min_data, max_data):
        return int(50 + 500 * (val - min_data) / (max_data - min_data + 1e-5))

    y_pos = 200
    min_x = map_val(min_val, min_val, max_val)
    q1_x = map_val(q1, min_val, max_val)
    med_x = map_val(median, min_val, max_val)
    q3_x = map_val(q3, min_val, max_val)
    max_x = map_val(max_val, min_val, max_val)

    # Draw box and lines
    cv2.line(canvas, (min_x, y_pos), (max_x, y_pos), (0,0,0), 2)
    cv2.rectangle(canvas, (q1_x, y_pos-20), (q3_x, y_pos+20), (255,0,0), 2)
    cv2.line(canvas, (med_x, y_pos-20), (med_x, y_pos+20), (0,0,255), 2)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'Min', (min_x-20, y_pos+40), font, 0.5, (0,0,0), 1)
    cv2.putText(canvas, 'Q1', (q1_x-20, y_pos-40), font, 0.5, (0,0,0), 1)
    cv2.putText(canvas, 'Median', (med_x-20, y_pos-40), font, 0.5, (0,0,0), 1)
    cv2.putText(canvas, 'Q3', (q3_x-20, y_pos-40), font, 0.5, (0,0,0), 1)
    cv2.putText(canvas, 'Max', (max_x-20, y_pos+40), font, 0.5, (0,0,0), 1)

    cv2.imwrite(save_path, canvas)

# ------------- MAIN PROCESS ------------- #

# Step 1: Preprocessing
print("Starting preprocessing...")
preprocessed_images = []
grayscale_images = []
filenames = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        filepath = os.path.join(INPUT_FOLDER, filename)
        img_color, img_gray = preprocess_image(filepath, apply_aug=APPLY_AUGMENTATION)

        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, (img_color * 255).astype(np.uint8))

        preprocessed_images.append(img_color)
        grayscale_images.append(img_gray)
        filenames.append(filename)

print(f"Preprocessed {len(preprocessed_images)} images.")

# Step 2: Cross-Correlation Analysis
print("Performing cross-correlation analysis...")
n = len(grayscale_images)
correlation_matrix = np.zeros((n, n), dtype=np.float32)

for i in range(n):
    for j in range(i, n):
        corr = cross_correlation_manual(grayscale_images[i], grayscale_images[j])
        correlation_matrix[i, j] = corr
        correlation_matrix[j, i] = corr  # symmetric matrix

create_heatmap(correlation_matrix, 'cross_correlation_heatmap.jpg')

# Step 3: Box Plot Analysis
print("Generating box plots...")
mean_intensities = []

for img_gray in grayscale_images:
    mean_intensity = np.mean(img_gray)
    mean_intensities.append(mean_intensity)

draw_boxplot(mean_intensities, 'mean_intensity_boxplot.jpg')

print("All tasks completed successfully.")






