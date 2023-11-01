from PIL import ImageOps


class AutoOrient:
    def __call__(self, image):
        # Use ImageOps.exif_transpose to automatically orient the image
        image = ImageOps.exif_transpose(image)
        return image
    

class ReduceImage:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image):
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
        print(type(image))
        width, height = image.size
        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        image = image.crop((left, top, right, bottom))
        return image