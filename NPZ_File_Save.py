import numpy as np
import os
import cv2

# Define the path to the two folders
# raw_folder = "D:/FYP/Dataset/Dataset for Image Enhancement/Raw"
# edited_folder = "D:/FYP/Dataset/Dataset for Image Enhancement/Edited"

raw_folder = "D:/FYP/Dataset/Dataset for Image Enhancement/01. Original Set/Raw"
edited_folder = "D:/FYP/Dataset/Dataset for Image Enhancement/01. Original Set/Edited"

WIDTH = 512
HEIGHT = 512

# Load the data from the two folders into two arrays
raw_images = []
for filename in os.listdir(raw_folder):
    file_path = os.path.join(raw_folder, filename)
    full_size_image_1 = cv2.imread(file_path)
    # raw_images.append(full_size_image_1,(WIDTH,HEIGHT))

    raw_images.append(cv2.resize(full_size_image_1,(WIDTH,HEIGHT),interpolation=cv2.INTER_CUBIC))
raw_images = np.array(raw_images)

edited_images = []
for filename in os.listdir(edited_folder):
    file_path = os.path.join(edited_folder, filename)
    full_size_image_2 = cv2.imread(file_path)
    # edited_images.append(full_size_image_2,(WIDTH,HEIGHT))

    edited_images.append(cv2.resize(full_size_image_2,(WIDTH,HEIGHT),interpolation=cv2.INTER_CUBIC))
edited_images = np.array(edited_images)


save_location = "D:/FYP/Phaedra_BE/imagedata2.npz"
# Save the two arrays into a npz file
# np.savez(save_location, raw_images = raw_images, edited_images = edited_images)
np.savez("imagedata2.npz", raw_images, edited_images)