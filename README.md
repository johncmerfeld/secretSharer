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

We will not explain every single line of code, but will try to elaborate on comments in the code so that the experiments can be more easily reproduced. We are primarily concerned with the secretSharerExp.py file and its utility library, secretUtils.py. Other utility functions that were used during development but not in the actual tests are located in extraUtils.py.

### 0. Experimental setup

After importing the necessary libraries and utility functions, we read hyperparameters from the command line.
```
numTrueSecrets = int(sys.argv[1])
numFalseSecrets = int(sys.argv[2])
numDistinctValues = int(sys.argv[3])
numEpochs = int(sys.argv[4])
batchSize = int(sys.argv[5])
```
Let's examine these one at a time. We are trying to extract the secret `s[r]` from the model. `numTrueSecrets` simply determines how many instances of `s[r]` we insert into the training data. `numFalseSecrets` determines how many instances of `s[r']` we insert into the data, where `s[r']` is some random secret of the same format as `s[r]`. `numDistinctValues` determines the randomness space of the secrets. Since every secret in the script has length 2, `numDistinctValues` determines how many different values each entry of the secret can take on. The randomness space is thus `numDistinctValues ^ 2`. `numEpochs` and `batchSize` will be given as training parameters to the model later, and we will discuss them more at that time.

The next block of code creates the secret according to hardcoded, but variable, specifications.
```
secretPref = "my locker combination is "
seqLength = len(secretPref.split())
gramSize = seqLength + 1

secretText = generateSecret(secretLength, numDistinctValues)
insertedSecret = secretPref + secretText
```

`gramSize` refers to the width of the **fixed-size windows** mentioned in the report, essentially the length of phrases the model reads in to make predictions off of. In this case, `gramSize` is 5, because we need the model to read the entire `secretPref` and produce a predicted value from it. The actual numeric value of the secret is computed randomly by the utility function `generateSecret()`.

### 1. Read data

The SMS messages are stored in XML format. Since XML is a nested markup language, we must traverse it like a tree to extract the right attributes.

```
root = ET.parse('smsCorpus_en_2015.03.09_all.xml').getroot()

d = []
for i in range(len(root)):
    d.append({'id' : root[i].get('id'),
              'text' : root[i][0].text})

rootId = len(root)
for i in range(numDistinctValues):
    a = str(i)
    d.append({'id' : rootId,
              'text' : gramSize * (a + " ")})
    rootId += 1

dataRaw = pd.DataFrame(d)
```

`root` is a large, nested object containing the message data. We structure that data with python's xml package and its `ElementTree` method. We then create a blank list, iterate through every value of `root`'s `text` attribute, and add it to the list. We then add strings of numbers to the list to ensure that they will appear in the dictionary (for potential secret values) even if they are not in the corpus. Finally, the text objects are saved as a Pandas DataFrame.

### 2. Clean data
