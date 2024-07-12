import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the directories and file paths
# TODO: image path
tiff_directory = '../C2 C4 Couplers'  # Directory containing the TIF files
score_file_path = './Plot/score_file.txt'  # Path to save the score file
template_path = './coupler.tif'  # Path to the template image

# If the best match score is above the threshold, crop and save the matched region
# TODO: threshold for bad devices, classified bad will not go to machine learning
threshold = 0.7

# Load the template image
template = cv2.imread(template_path, 0)

# List to store the best match scores
best_scores = []

# Iterate through each TIF file in the directory
for tiff_file in os.listdir(tiff_directory):
    if tiff_file.endswith('.tif'):
        original_path = os.path.join(tiff_directory, tiff_file)

        # Load the TIF image in grayscale
        img = cv2.imread(original_path, 0)
        # Load the TIF image in color
        image = cv2.imread(original_path, 1)

        # Get the dimensions of the template image
        w, h = template.shape[::-1]

        # Perform template matching
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        # Find the best match location and score
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        best_scores.append(max_val)

        print(f"The best match score is : {max_val}, image name is {tiff_file}")

        # Save the best match score to the score file
        with open(score_file_path, 'a') as f:
            f.write(f"{tiff_file} {max_val}\n")


        if max_val >= threshold:
            x, y = max_loc
            cropped_img = image[y:y + h, x:x + w]
            cv2.imwrite(f'./template/{tiff_file}', cropped_img)

# Divide the scores into two groups based on the threshold
low_scores = [score for score in best_scores if score < threshold]
high_scores = [score for score in best_scores if score >= threshold]

# Plot the histogram of best match scores
plt.hist(low_scores, bins=20, color='red', edgecolor='black', alpha=0.7, label='Score < 0.7')
plt.hist(high_scores, bins=20, color='green', edgecolor='black', alpha=0.7, label='Score >= 0.7')

# Add a vertical line at the threshold
plt.axvline(x=threshold, color='blue', linestyle='--', label='Threshold = 0.7')
plt.title('Histogram of Best Match Scores')
plt.xlabel('Match Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('best_match_scores_histogram.png')
plt.show()
