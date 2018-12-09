# Secret sharer and other words

This is the technical writeup for my Fall 2018 final project in CS 591 at Boston University. The accompanying code and report are in this repository. The rest of this text will assume familiarity with the [2018 Carlini et al. paper](https://arxiv.org/abs/1802.08232) on which this work is based, as well as the attached FinalProjectReport.pdf.

## Tech specs
  - **Environment**: Python 3.7.0; TensorFlow 1.12.0, run on a Macbook Pro CPU; Keras 2.2.4; Pandas 0.23.4; NumPy 1.15.4;
  - **Summary of typical model**:
  ```______________________________________________________________________
Layer (type)                 Output Shape                     Param #   
======================================================================
Embedding          (None, 4, 4)                             56336 (number of word groups in training data)
______________________________________________________________________
LSTM               (None, 4, 100)                           42000     
______________________________________________________________________
LSTM               (None, 100)                              80400     
______________________________________________________________________
Dense              (None, 100)                              10100     
______________________________________________________________________
Dense             (None, 14084 (number of distinct words))  1422484   
======================================================================
Total params: 1,611,320
______________________________________________________________________
```

## Differences from Carlini paper
Apart from different model architectures, there are two primary differences between this code and the procedure described by Carlini. One is the fixed-size versus stateful processing of the text. The main results of the Carilni paper used a recurrent neural network to process arbitrary-length text input, whereas this report used fixed-size partioning as its main strategy for turning the source text into data. This affected the way models could be attacked to extract secrets, because our model could only give predictions on certain tokens of input; a different-length secret and prefix necessitated an entirely different model.
The second difference was in how we estimated exposure. Due to technical constraints, we focussed on relatively narrow randomness spaces so that entire secret phrases' probabilities could be enumerated over, whereas the Carlini paper developed efficient ways of searching over large randomness spaces to extract the secret one character at a time. We believe that despite our more limited methodology, our results offer meaningful insights and a validation of the original results.

## Code walkthrough
