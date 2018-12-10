# Using the Secret Sharer Attack to  Learn Locker Combinations from Text Messages
## John C. Merfeld, Boston University

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

We will not explain every single line of code (i.e. you will not get correct results if you simply copy and paste the code snippets from this file), but will try to elaborate on comments in the code so that the experiments can be more easily reproduced. We are primarily concerned with the secretSharerExp.py file and its utility library, secretUtils.py. Other utility functions that were used during development but not in the actual tests are located in extraUtils.py.

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

We create new columns in our DataFrame to save cleaner versions of the text. It must be scrubbed of unhelpful punctuation and non-standard spellings.
```
myPunc = '!"#$%&\()*+-/:;<=>?@[\\]^_`{|}~\''
dataRaw['noPunc'] = dataRaw['text'].apply(
        lambda s: s.translate(str.maketrans('','', myPunc)).lower()
        )

dataRaw['splchk'] = dataRaw['noPunc'].apply(cleanSMS)
```

We define a list of punctuation characters to be removed by the `apply()` function. Then, the final column ("spellcheck") is created by applying the `cleanSMS()` utility function. It is worth reading some of the source code of this function, as it contains over 200 regular expression commands for standardizing the language of the corpus.

```
mskTrain = np.random.rand(len(dataRaw)) < 0.8
dataRawR = dataRaw[mskTrain]
dataRawT = dataRaw[~mskTrain]

mskVal = np.random.rand(len(dataRawR)) < 0.8
dataRawV = dataRawR[~mskVal]
dataRawR = dataRawR[mskVal]
```
We then select 20% of the rows to be set aside as a test set and 20% of what remains as a validation set to monitor the training process. The remaining 64% of records are for training the. The reason this partitioning happens so early is so that we can insert the secret into the training set deliberately and before the messages are split into 5-grams.

```
d, rootId = enumerateSecrets(secretLength, numDistinctValues, rootId, secretPref)

if numFalseSecrets > 0:
    noise = [d[i] for i in sorted(random.sample(range(len(d)), numFalseSecrets))]
    noiseDF = pd.DataFrame(noise)

testSecret = pd.DataFrame(d);
dataRawT = dataRawT.append(d)

d = []
for i in range(numTrueSecrets):
    d.append({'id' : rootId,
              'text' : insertedSecret,
              'noPunc' : insertedSecret,
              'splchk' : insertedSecret})
    rootId += 1

trainSecret = pd.DataFrame(d)
dataRawR = dataRawR.append(d)
if numFalseSecrets > 0:
    dataRawR = dataRawR.append(noiseDF)
```

The `enumerateSecrets()` utility function allows us to add every secret permutation to the "test" data set, allowing us to see what probability the model assigns to every value of `r`. We also see in this step how `numTrueSecrets` instances of `s[r]` are added to the training data, along with the noisy `s[r']`.

```
d = []
gid = 0
for i in range(len(dataRawR)):
    grams = ngrams(dataRawR.splchk.iloc[i].split(), gramSize)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGramsR = pd.DataFrame(d)
```

Here, the ngrams package is used to create new 5-word tuples as the data entires. WHen the model is trained, the first four words will act as the data, and the final one will be the label. This is repeated for the testing and validation sets.

### 3. Create dictionary

First, we create two dictionaries, with words as keys. The first has unique numeric IDs, the second counts word frequencies.

```
dct = dict()
dctFreq = dict()

did = 0
for i in range(len(dataRaw)):
    s = dataRaw.splchk.iloc[i].split()
    for w in s:
        if w not in dct:
            dct[w] = did
            did += 1
            dctFreq[w] = 1
        else:
            dctFreq[w] += 1
```

Now, we create a third dictionary and load in all the words with a frequency greater than one. This is more effective than simply deleting words form the first dictionary because it lowers the ID values, which will reduce the data size later when we one-hot encode the records.

```
dctNoSingle = dict()
did = 0
for w in list(dct.keys()):
    if dctFreq[w] != 1:
        dctNoSingle[w] = did
        did += 1

dct = dctNoSingle

def noSingleUseWords(tup):
    for w in tup:
        if w not in dct:
            return False
    return True

dataGramsR = dataGramsR[dataGramsR['data'].apply(noSingleUseWords) == True]
```

### 4. Transform data

Now that we have our dictionary, we simply replace each word with its numeric value.

```
def encodeText(tup):
    code = [None] * len(tup)
    for i in range(len(tup)):
        code[i] = dct[tup[i]]  
    return tuple(code)

dataGramsR['codes'] = dataGramsR['data'].apply(encodeText)

```

All steps are repeated for testing and validation, of course. Now we simply split our strings of numbers into a "data" and "label" portion (`x` and `y`), and reform them as numpy arrays.

```
dataGramsR['x'] = dataGramsR['codes'].apply(dataSplit)
dataGramsR['y'] = dataGramsR['codes'].apply(labelSplit)

xr = np.zeros((len(dataGramsR), seqLength), dtype = int)
yr = np.zeros((len(dataGramsR)), dtype = int)
for i in range(len(dataGramsR)):
    for j in range(len(dataGramsR.x.iloc[i])):
        xr[i][j] = dataGramsR.x.iloc[i][j]
    yr[i] = dataGramsR.y.iloc[i]
```

### 5. Train model
To have the shapes of data correct within the nodes of the model, we must one-hot encode the label array. We do the same for the validation label.

```
vocabSize = len(dct)
b = np.zeros((len(yr), vocabSize))
b[np.arange(len(yr)), yr] = 1

```

We build the model with 5 layers: the initial embedding of the data, two Long Short Term Memory (LSTM) layers with 100 nodes each, and 2 Dense layers to do the multinomial classification on.

When the model is training, the `epochs` parameter refers to how many passes of the data will be performed, and `batch_size` refers to how many records will be run at a time during each pass. A higher batch size and fewer epochs generally mean the model trains faster.

```
model = Sequential()
model.add(Embedding(vocabSize, seqLength, input_length = seqLength))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(vocabSize, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

history = model.fit(xr, b, batch_size = batchSize, epochs = numEpochs, verbose = True,
                    validation_data = (xv, bv))

```

### 6. Calculate exposure

Now, we feed the prefix and every possible first (of two) entries for the secret. That way, we get probabilities for each first secret entry, and conditional probabilities for the entire secret. The `numericProbs()` utility function is used here; it returns an array of the model's scores of every entry in the dictionary that represents a number in the randomness space.

```
start = len(xt) - secretLength * (numDistinctValues ** secretLength)

p0 = np.ones((numDistinctValues, numDistinctValues), dtype = float)
for i in range(start, len(xt), 2 * numDistinctValues):
    k = int((i-start) / (2 * numDistinctValues))
    p0[k] = numericProbs(xt, numDistinctValues, dct, seqLength, model, i)
    p1 = numericProbs(xt, numDistinctValues, dct, seqLength, model, i + 1)
    p0[k] = p0[k][k] * p1

scoresRaw = np.argsort(p0, None)[::-1]

```

Once we have scores for every secret, we calculate their rank and exposure as defined in the report, then append the exposure, along with the experimental parameters, to our analysis file.

```
d = []
for i in range(len(scoresRaw)):
    d.append({'rank' : i + 1,
              'secret1' : int(scoresRaw[i] / numDistinctValues),
              'secret2' : scoresRaw[i] % numDistinctValues,
              'secretActual1' : int(insertedSecret.split()[-2]),
              'secretActual2' : int(insertedSecret.split()[-1])})

secretRanks = pd.DataFrame(d)
secretMatch1 = secretRanks[secretRanks.secret1 == secretRanks.secretActual1]
secretMatch2 = int(secretMatch1[secretMatch1.secret2 == secretMatch1.secretActual2]['rank'])

exposure = log(bigR, 2) - log(secretMatch, 2)
d = []
d.append({'numEpochs' : numEpochs,
          'batchSize' : batchSize,
          'numTrueSecrets' : numTrueSecrets,
          'numFalseSecrets' : numFalseSecrets,
          'randomnessSpace' : numDistinctValues,
          'secretPrefixLength' : seqLength,
          'secretType' : secretPref,
          'exposure': exposure})

results = pd.DataFrame(d)

fileName = "experimentalResults.csv"
if not os.path.isfile(fileName):
   results.to_csv(fileName, sep = ',', index = False)
else:
   results.to_csv(fileName, mode = 'a', sep = ',', header = False, index = False)

```
And that's it! This script was run over 100 times to generate the data for the report; over 2 full days on a Mac CPU.

Ideally, the code would accommodate more complicated secrets, especially those beyond the numeric space. But at least in this iteration, there simply wasn't time to figure out the technical details on that front for this iteration.
