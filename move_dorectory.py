import os
import shutil

# Define folder paths
folder1_path = './label/bad'  # Path to the first folder
folder2_path = './template_jpg'  # Path to the second folder
destination_folder = './newLabel/bad'  # Path to the destination folder

# Collect all filenames from folder1 into a set for quick lookup
file_names_set = set(os.listdir(folder1_path))

# Iterate through the files in folder2
for filename in os.listdir(folder2_path):
    # Check if the filename is in the set of filenames from folder1
    if filename in file_names_set:
        # Construct full file paths
        source_file_path = os.path.join(folder2_path, filename)
        destination_file_path = os.path.join(destination_folder, filename)

        # Move the file to the destination folder
        try:
            shutil.move(source_file_path, destination_file_path)
            print(f"Moved '{filename}' to '{destination_folder}'")
        except Exception as e:
            print(f"Error moving '{filename}': {e}")

print("Operation completed.")
