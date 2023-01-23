# Python Perceptron

A perceptron is a type of artificial neural network that is used for binary classification. It consists of a single layer of perceptrons, which are simple processing units that take in input from multiple sources, combine it with a set of weights, and produce a single output. The output is a binary value (i.e., 0 or 1) that represents the class to which the input belongs. The perceptrons in a single layer are connected to each other and to the input sources. 

The weights of the connections between the perceptrons and the input sources are adjusted during training, so that the perceptron can learn to correctly classify the input data. The perceptron algorithm is a linear classifier, which means that it is only able to classify linearly separable data. However, it can be extended to multi-layer perceptrons, which are able to classify non-linearly separable data.

The original paper written by Frank Rosenblatt ia available [here](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=65a83117cbcc4e6eb7c6ac5be8e61195dc84b9fc).

## Prerequisites
This project is implemented with Python 3 and the dependencies found in `requirements.txt`. The level of Python knowledge required is intermediate, yet I did my best to keep the syntaax as simple as possible.

## Running the Perceptron
The script consists of three classes:
- a Generator that generates trainig material based on the fonts available on your system
- a Processor that transformes images into arrays
- a Perceptron that predicts results upon training

The commands you can use to generate a new learning set are:
```py
# Generate new images
Generator().generate_training_list(<PATH_TO_FOLDER>)

# Generate training set
TRAINING_SET_A = [Processor().get_simplified_array(Processor().get_pixels_array(Processor().read_image(path=f'<PATH_TO_LETTER>/{i}'), <PIXELS_W>, <PIXELS_H>)) for i in os.listdir('<PATH_TO_FOLDER>')]
```

The command to Process a image is:
```py
Processor().get_simplified_array(Processor().get_pixels_array(Processor().read_image(path=f'<PATH_TO_LETTER>'), <PIXELS_W>, <PIXELS_H>))
```

The commands to train and predict are:
```py
PERCEPTRON = Perceptron(letter=<LETTER>)
PERCEPTRON.train(<TRAINING_SET>)
PERCEPTRON.predict(Processor().get_simplified_array(Processor().get_pixels_array(Processor().read_image(path=f'<PATH_TO_LETTER>'), <PIXELS_W>, <PIXELS_H>)))
```

This project is authored by [***Mihnea Manolache***](https://github.com/mihneamanolache) and guided by [***Conf. univ. dr. Popescu-Bodorin Nicolaie***](https://www.linkedin.com/in/bodorin/). 
