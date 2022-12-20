from matplotlib import font_manager
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
import io, string, os, math
import numpy as np

''' ------------------------------------------------ PERCEPTRON ------------------------------------------------ '''


class Perceptron:

    def __init__(self) -> None:
        self.w_sum: int = 0
        self.weigths: list = []
        self.bias: float = 0.1

    ''' ----- Training data generation section  ----- '''

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
        font = ImageFont.truetype(font, int(w_h/1.75))
        _, _, w, h = draw.textbbox((0, 0), letter, font=font)
        draw.text(((w_h-w)/2, (w_h-h)/2), letter, font=font)
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

    ''' ----- General methods for reading images and returning bytes arrays ----- '''

    # Read image from patch and convert it to BW
    def read_image(self, path: str) -> Image:
        image: Image = Image.open(path)
        image = image.convert('1')
        image.save(path)
        return image

    # Get all image bytes
    def image_to_byte_array(self, image: Image) -> bytes:
        byteIO: bytes = io.BytesIO()
        image.save(byteIO, format='JPEG')
        return byteIO.getvalue()
    
    # Get pixel color at given (x,y) location
    def get_pixel(self, img: Image, x: int, y: int) -> int:
        return img.getpixel((x, y))

    # Generate list of arrays for each line of the image
    def get_pixels_array(self, img: Image, width: int, height:int) -> list:
        arr: list = []
        for i in range(width):
            b = [self.get_pixel(img, i, j) for j in range(height)]
            arr.append(b)
        return arr
    
    ''' ----- Perceptor Algorythm Implementation ----- '''

    def step(self, w_sum: float) -> int:
        return np.where(w_sum >= 0.0, 1, 0)

    def training() -> None:
        '''
        NOTES:
            - Should itterate over value_input, weigth in zip(trainig_set, self.weights)
                - Add value_input * weigth to self.w_sum
            - retrun self.step(self.w_sum)

        EXAMPLE:
            for x, w in zip(trainig_set, self.weights):
                self.w_sum += x*w + self.bias
            return self.step(self.w_sum)
        '''
        return


''' ------------------------------------------------ EXECUTION ------------------------------------------------ '''


''' 
STEP 1: Create a new perceptron 
    USAGE: <VARIABLE> = Perceptron()
        - i.e. perceptron = Perceptron()
    NOTES: 
        - Design wise, I think I will generate a new Perceptron object for each letter
        and train it accordingly, such that it will return 1 or 0 if the user input 
        belongs or not to the trained object (letter).
'''
perceptron = Perceptron()



''' 
STEP 2 (OPTIONAL): Generate a new set of images for training 
    USAGE: <VARIABLE>.generate_training_list(<PATH_TO_FOLDER>)
        - i.e. perceptron.generate_training_list('./letters')
'''
# perceptron.generate_training_list('./letters')

'''
STEP 3: Model training
    NOTES: 
        - The system should loop through every image of one character in the traning set
        - It should get the pixels array (`get_pixels_array`) which is composed of arrays 
        pixels bytes, for each line of the image
        - If the multiplied elements of the arrays inside the pixels array is 0, it means 
        it holds a black pixel (0 in RGB is black, 255 is white) and we'll use only these lines
        - We'll count the black pixels inside the array and divide by image width to get 
        black pixels ratio for each line

    EXAMPLE:
        image = perceptron.read_image('A.jpeg')
        print([line.count(0)/len(line)*100 for line in perceptron.get_pixels_array(image, image.width, image.height) if math.prod(line) == 0])

        UNCOMPREHENDED (FOR READABILITY):
            for letter in os.listdir('./letters/a'):
                arr = []
                image = perceptron.read_image(letter)
                for line in perceptron.get_pixels_array(image, image.width, image.height):
                    if math.prod(line) == 0:
                        arr.push(line.count(0)/len(line)*100)
                print(arr)

        - Weights can be generated using `np.zeros(math.prod(image.size))`
        - Should adjust the weigths based on that output. Will ask professor !!!
'''


'''
STEP 4: Prediction based on user input
    NOTES: 
        - I will probably encapsulate everything in a `while True` loop to run the
        program continously
        - I will probably require user input, which will be of type string and its
        value will be the path to an image containing a letter
        - I will get the image's bytes ratio array and feed it to the trained models
        like in the example from STEP 3.
'''