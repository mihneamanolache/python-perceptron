from matplotlib import font_manager
from PIL import Image, ImageFont, ImageDraw, ImageChops
from fontTools.ttLib import TTFont
import io, string, os, math
import numpy as np
from fontTools import ttLib
import cv2


''' ------------------------------------------------ PERCEPTRON ------------------------------------------------ '''

class Generator:
    def __init__(self) -> None:
        self.name: string = 'Generator'

    # Check if glyph is available for given font and character
    def has_glyph(self, font: string, glyph: string) -> bool:
        for table in font['cmap'].tables:
            if table.isUnicode() or table.getEncoding() == 'utf_16_be':
                if ord(glyph) in table.cmap.keys():
                    return True
        return False 

    # Draw new BW image with character in the middle
    def generate_image(self, w_h: int, font: str, letter: str, path: str, i: int) -> None:
        image = Image.new("1", (w_h, w_h), "white")
        draw = ImageDraw.Draw(image)
        font_size = int(w_h/1.75)
        f = ImageFont.truetype(font, font_size)
        _, _, w, h = draw.textbbox((0, 0), letter, font=f)
        draw.text(((w_h-w)/2, (w_h-h)/2), letter, font=f)
        image.save(f'{path}/{i}.jpeg')

    # Generate list of alphabetical characters
    def generate_alphabet(self) -> list:
        return [i for i in string.ascii_lowercase] + [i for i in string.ascii_uppercase]

    # Get all available fonts of type ttf
    def get_system_fonts(self) -> list:
        return font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    # Generste training images
    def generate_training_list(self, path: str) -> None:
        os.rmdir(path)
        os.mkdir(path)
        for letter in self.generate_alphabet():
            i = 0
            _path = f'./{path}/{letter}' if not os.path.exists(f'./{path}/{letter}') else f'./{path}/_{letter}'
            os.makedirs(_path)
            for font in self.get_system_fonts():
                if 'ttf' in font:
                    if self.has_glyph(TTFont(font), letter):
                        self.generate_image(200, font, letter, _path, i)
                        i+=1


class Processor:
    def __init__(self) -> None:
        self.name: string = 'Pre-processor'

    # Crop whitespace from image
    def crop_white(self, image: str, pixel: int = 255) -> None:
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        crop_rows = gray[~np.all(gray == pixel, axis=1), :]
        cropped_image = crop_rows[:, ~np.all(crop_rows == pixel, axis=0)]
        cv2.imwrite(image, cropped_image)   

    # Read image from path and convert it to BW
    def read_image(self, path: str) -> Image:
        print('[i] Reading image.')
        self.crop_white(path)
        image: Image = Image.open(path)
        image = image.convert('1')
        ratio = 200/image.size[1]
        resized = image.resize((int(ratio*image.size[0]),200))
        resized.save(path)
        return resized

    # Get all image bytes
    def image_to_byte_array(self, image: Image) -> bytes:
        print('[i] Reading image bytes.')
        byteIO: bytes = io.BytesIO()
        image.save(byteIO, format='JPEG')
        return byteIO.getvalue()
    
    # Get pixel color at given (x,y) location
    def get_pixel(self, img: Image, x: int, y: int) -> int:
        return img.getpixel((x, y))

    # Generate list of arrays for each line of the image
    def get_pixels_array(self, img: Image, width: int, height:int) -> list:
        print('[i] Gennerating array of bytes.')
        arr: list = []
        for i in range(width):
            b = [self.get_pixel(img, i, j) for j in range(height)]
            arr.append(b)
        return arr
    
    def get_simplified_array(self, array: list) -> list:
        print('[i] Generating input vector.')
        return [line.count(0)/len(line)*100 for line in array if math.prod(line) == 0]


class Perceptron:
    def __init__(self, letter: str) -> None:
        self.letter: str = letter
        self.w_sum: int = 0
        self.weights: list = []
        self.bias: float = 0.1
    
    def activation(self, w_sum: float) -> int:
        return np.where(w_sum >= 0.0, 1, 0) 

    def train(self, training_set: list) -> None:
        print(f'[i] Training perceptron for letter {self.letter}')
        for training_sequence in training_set:
            for x, w in zip(training_sequence, self.weights):
                    self.w_sum += x * w + self.bias
                    if self.activation(self.w_sum) < 1:
                        self.weights = [self.weights[i] * training_sequence[i] + self.bias for i in range(0, len(self.weights))]
           
    def predict(self, vector: list) ->  None:
        for x, w in zip(vector, self.weights):
            self.w_sum += x * w + self.bias
        return self.activation(self.w_sum)


''' ------------------------------------------------ EXECUTION ------------------------------------------------ '''

''' 
(OPTIONAL): Generate a new set of images for training 
    USAGE: Generator().generate_training_list(<PATH_TO_FOLDER>)
'''
# Generator().generate_training_list('./letters')


'''
STEP 1: Generate training set (inputs)
    USAGE: TRAINING_SET_A = [Processor().get_simplified_array(Processor().get_pixels_array(Processor().read_image(path=f'<PATH_TO_LETTER>/{i}'), <PIXELS_W>, <PIXELS_H>)) for i in os.listdir('<PATH_TO_FOLDER>')] 
'''
TRAINING_SET_A = [Processor().get_simplified_array(Processor().get_pixels_array(Processor().read_image(path=f'./letters/_A/{i}'), 200, 200)) for i in os.listdir('./letters/_A')] 
# print(len(Processor().get_simplified_array(Processor().get_pixels_array(Processor().read_image(path=f'./letters/_A/2.jpeg'), 200, 200))))


''' 
STEP 2: Initialize perceptron 
    USAGE: PERCEPTRON_<LETTER> = Perceptron(letter=<LETTER>)
'''
PERCEPTRON_A = Perceptron(letter='A')

'''
STEP 3: Model training
    USAGE: PERCEPTRON_<LETTER>.train(<TRAINING_SET>)
'''
PERCEPTRON_A.train(TRAINING_SET_A)


'''
STEP 4: Prediction
    USAGE: PERCEPTRON_<LETTER>.predict(Processor().get_simplified_array(Processor().get_pixels_array(Processor().read_image(path=f'<PATH_TO_LETTER>'), <PIXELS_W>, <PIXELS_H>)))
'''
print(PERCEPTRON_A.predict(Processor().get_simplified_array(Processor().get_pixels_array(Processor().read_image(path=f'A.jpeg'), 200, 200))))