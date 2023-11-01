from PIL import ImageOps

class AutoOrient:
    def __call__(self, image):
        """
        Automatically orients the image based on EXIF data using ImageOps.exif_transpose.
        
        Args:
            image (PIL.Image.Image): The input image.
        
        Returns:
            PIL.Image.Image: The oriented image.
        """
        # Use ImageOps.exif_transpose to automatically orient the image
        image = ImageOps.exif_transpose(image)
        return image

class ReduceImage:
    def __init__(self, scale):
        """
        Initializes the ReduceImage transform with a scale factor.

        Args:
            scale (float): The scale factor to resize the image (e.g., 0.5 for 50% reduction).
        """
        self.scale = scale

    def __call__(self, image):
        """
        Reduces the image size by the specified scale factor.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The resized image.
        """
        width, height = image.size
        new_width = int(width * self.scale)
        new_height = int(height * self.scale)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        image = image.crop((left, top, right, bottom))
        return image

class MakeSquare:
    def __call__(self, image):
        """
        Makes the input image square by cropping it.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The squared image.
        """
        width, height = image.size
        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        image = image.crop((left, top, right, bottom))
        return image
