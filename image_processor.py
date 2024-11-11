from PIL import Image, ImageOps
import numpy as np

class ImageProcessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def preprocess_image(self, image_file):
        # Open the image with PIL and convert to RGB
        image = Image.open(image_file).convert("RGB")
        # Resize the image to the target size
        image = ImageOps.fit(image, self.target_size, method=Image.Resampling.BILINEAR, bleed=0.0, centering=(0.5, 0.5))
        # Convert the image to a numpy array and normalize to [0, 1]
        img_array = np.array(image) / 255.0
        # Expand dimensions to add batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # Shape (1, height, width, 3)
        return img_array

    def postprocess_mask(self, mask_array):
        # Remove the batch dimension
        mask_array = np.squeeze(mask_array, axis=0)
        # If the mask has multiple channels, pick the channel with the highest probability
        if mask_array.shape[-1] > 1:
            mask_array = np.argmax(mask_array, axis=-1)
        # Convert the mask to binary (0 and 1) based on a threshold, if necessary
        mask_array = (mask_array > 0.5).astype(np.uint8) * 255  # Convert to 0 and 255 for visualization
        return mask_array
