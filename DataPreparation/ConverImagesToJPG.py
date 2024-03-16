import os

from PIL.Image import Image


def to_jpg(root_folder, output_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file is an image
            if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                try:
                    # Determine the relative path for the output folder
                    relative_path = os.path.relpath(root, root_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)

                    # Create the corresponding subfolder in the output directory
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)

                    # Open and convert the image to JPG format
                    with Image.open(file_path) as img:
                        img = img.convert("RGB")  # Convert to RGB mode if needed
                        output_file = os.path.join(output_subfolder, os.path.splitext(file)[0] + '.jpg')
                        img.save(output_file, 'JPEG')
                        print(f"Converted {file_path} to JPG format and saved as {output_file}")
                except Exception as e:
                    print(f"Error converting {file_path}: {e}")


if __name__ == "__main__":
    root_folder = "BIRDS"  # Specify the root folder containing images
    output_folder = "BIRDS1"  # Specify the output folder for converted images

    to_jpg(root_folder, output_folder)

