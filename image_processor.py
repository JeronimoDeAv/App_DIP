from PIL import Image, ImageOps
import numpy as np

class ImageProcessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def preprocess_image(self, image_file, color_mode="grayscale"):
        # Abrir la imagen en el modo correspondiente
        if color_mode == "grayscale":
            image = Image.open(image_file).convert("L")  # Modo "L" para escala de grises
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)  # Añadir dimensión de canal
        elif color_mode == "rgb":
            image = Image.open(image_file).convert("RGB")  # Modo "RGB"
            img_array = np.array(image) / 255.0  # Normalizar a [0, 1]

        # Cambiar tamaño y expandir dimensiones
        image = ImageOps.fit(image, self.target_size, method=Image.Resampling.BILINEAR)
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de lote
        return img_array

    def postprocess_mask(self, mask_array):
        # Remover la dimensión de lote y binarizar la máscara
        mask_array = np.squeeze(mask_array, axis=0)
        mask_array = (mask_array > 0.5).astype(np.uint8) * 255
        return mask_array
