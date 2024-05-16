import os
import cv2


def resize_images(input_folder, output_folder, target_width=1920):


    # Iterate through all images in the input folder and its subfolders
    for root, _, files in os.walk(input_folder):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif')):
                # Read the image
                image_path = os.path.join(root, file)
                print(image_path)
                image = cv2.imread(image_path)

                # Calculate aspect ratio
                height, width = image.shape[:2]
                aspect_ratio = width / height

                # Calculate new height
                new_height = int(target_width / aspect_ratio)

                # Resize the image
                resized_image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)

                # Create subfolders in the output folder if necessary
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # Save the resized image
                output_path = os.path.join(output_subfolder, file)
                cv2.imwrite(output_path, resized_image)


# Example usage:
input_folder = 'data/Panda/images/Det'
output_folder = 'data/Panda/images/Det_Smaller'
resize_images(input_folder, output_folder)
