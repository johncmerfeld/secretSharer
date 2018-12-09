# Secret sharer and other words

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
There are two primary differences between this code and the procedure described in the [2018 Carlini et al. paper](https://arxiv.org/abs/1802.08232). 

## Code walkthrough
