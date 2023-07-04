from PIL import Image
import os


input_folders = ["img_align_celeba"]

output_size = (64, 64)

for input_folder in input_folders:
    # Delete the Thumbs.db file if it exists
    thumbs_db_path = os.path.join(input_folder, "Thumbs.db")
    if os.path.exists(thumbs_db_path):
        os.remove(thumbs_db_path)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if it is a file (and not a directory)
        if os.path.isfile(file_path):
            try:
                # Open the image
                with Image.open(file_path) as im:
                    # Verify the image
                    im.verify()

                # Check if the image size matches the target size
                im = Image.open(file_path)
                image_size = im.size
                if image_size != output_size:
                    # Resize the image
                    im_resized = im.resize(output_size, Image.ANTIALIAS)

                    # Save the resized image back to the original file
                    im_resized.save(file_path)

            except:
                # If an error is thrown, it's probably a bad image, delete it
                print(f"Deleting corrupted image: {file_path}")
                os.remove(file_path)

print("Images have been processed")

# import numpy as np

# image_path = "img_align_celeba/105135.jpg"

# try:
#     image = Image.open(image_path)
#     image_array = np.array(image)
#     print("Image loaded successfully.")
#     print("Image array shape:", image_array.shape)
#     print("Image array content:\n", image_array)
# except (IOError, OSError) as e:
#     print(f"Error loading the image: {e}")
    