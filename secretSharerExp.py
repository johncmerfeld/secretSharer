import sys, os, random
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from nltk import ngrams
from math import log

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

from secretUtils import cleanSMS, dataSplit, labelSplit, generateSecret
from secretUtils import comboString, enumerateSecrets, numericProbs

# applied function to remove data entries containing words only seen once in
#   the entire corpus
def noSingleUseWords(tup):
    for w in tup:
        if w not in dct:
            return False
    return True

# applied function to encode words from the corpus as unique numeric values
def encodeText(tup):
    code = [None] * len(tup)
    for i in range(len(tup)):
        code[i] = dct[tup[i]]  
    return tuple(code)

# 0. EXPERIMENTAL SETUP ====================================

# how many copies of the secret do we insert?
numTrueSecrets = int(sys.argv[1])
# how many 'noisy' secrets do we insert?
numFalseSecrets = int(sys.argv[2])
# how many ticks are on our lock?
numDistinctValues = int(sys.argv[3])
# how long should we train the model?
numEpochs = int(sys.argv[4])
batchSize = int(sys.argv[5])

# what form should the secret take?
secretPref = "my locker combination is "
seqLength = len(secretPref.split())
gramSize = seqLength + 1

# randomness space
secretLength = 2
bigR = numDistinctValues ** secretLength

# generate a random secret
secretText = generateSecret(secretLength, numDistinctValues)
insertedSecret = secretPref + secretText

print("\n+---------------------------------------+")
print("| THANK YOU FOR USING THE SECRET SHARER |")
print("+---------------------------------------+\n")
print(" True secrets inserted:", numTrueSecrets)
print(" False secrets inserted:", numFalseSecrets)
print(" Randomness space:", numDistinctValues)
print(" Training epochs:", numEpochs)
print(" Batch size:", batchSize)
print(" Secret text: '", insertedSecret, "'\n", sep = '')
print("-----------------------------------------")
print("\npreparing data...")

# 1. READ DATA =============================================

# 1.1 PARSE XML --------------------------------------------
root = ET.parse('smsCorpus_en_2015.03.09_all.xml').getroot()

d = []
for i in range(len(root)):
    d.append({'id' : root[i].get('id'),
              'text' : root[i][0].text})

# 1.2 ADD NUMBERS TO THE VOCABULARY ------------------------
rootId = len(root)
for i in range(numDistinctValues):
    a = comboString(i)
    d.append({'id' : rootId,
              'text' : gramSize * (a + " ")})
    rootId += 1
    
dataRaw = pd.DataFrame(d)

# 2. CLEAN DATA ============================================

# 2.1 REMOVE PUNCTUATION AND MAKE LOWER CASE ---------------
myPunc = '!"#$%&\()*+-/:;<=>?@[\\]^_`{|}~\''
dataRaw['noPunc'] = dataRaw['text'].apply(
        lambda s: s.translate(str.maketrans('','', myPunc)).lower()
        )

# 2.2 SCRUB MESSAGES ----------------------------------------   
# found that this needed to be done twice to find words separated
#   by the first iteration
dataRaw['splchk'] = dataRaw['noPunc'].apply(cleanSMS)
dataRaw['splchk'] = dataRaw['splchk'].apply(cleanSMS)

# 2.2 SPLIT INTO TRAIN, TEST, AND VALIDATION ---------------
# train-test split
mskTrain = np.random.rand(len(dataRaw)) < 0.8
dataRawR = dataRaw[mskTrain]
dataRawT = dataRaw[~mskTrain]

# train-validation split
mskVal = np.random.rand(len(dataRawR)) < 0.8
dataRawV = dataRawR[~mskVal]
dataRawR = dataRawR[mskVal]

# 2.3 INSERT SECRET ---------------------------------------
# add all possible secrets to the test data for exposure calculations later
d, rootId = enumerateSecrets(secretLength, numDistinctValues, rootId, secretPref)

# get some noise from these fake secret to add to training
if numFalseSecrets > 0:
    noise = [d[i] for i in sorted(random.sample(range(len(d)), numFalseSecrets))]
    noiseDF = pd.DataFrame(noise)

testSecret = pd.DataFrame(d);
dataRawT = dataRawT.append(d)

d = []
# several in training data
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

# 2.4 SPLIT INTO OVERLAPPING SETS OF WORDS -----------000000

d = []
gid = 0
for i in range(len(dataRawR)):
    grams = ngrams(dataRawR.splchk.iloc[i].split(), gramSize)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGramsR = pd.DataFrame(d)

d = []
for i in range(len(dataRawV)):
    grams = ngrams(dataRawV.splchk.iloc[i].split(), gramSize)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGramsV = pd.DataFrame(d)

d = []
for i in range(len(dataRawT)):
    grams = ngrams(dataRawT.splchk.iloc[i].split(), gramSize)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGramsT = pd.DataFrame(d)

# 3. CREATE DICTIONARY =====================================

# 3.1 CREATE DICTIONARY OF UNIQUE WORDS --------------------
# word IDs
dct = dict()
# word frequencies
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

# 3.2 REMOVE SINGLE-USE WORDS FROM DICTIONARY --------------
dctNoSingle = dict()
did = 0
for w in list(dct.keys()):
    if dctFreq[w] != 1:
        dctNoSingle[w] = did
        did += 1
        
dct = dctNoSingle

# 3.3 REMOVE NGRAMS WITH SINGLE-USE WORDS FROM DATA --------
dataGramsR = dataGramsR[dataGramsR['data'].apply(noSingleUseWords) == True]
dataGramsT = dataGramsT[dataGramsT['data'].apply(noSingleUseWords) == True]
dataGramsV = dataGramsV[dataGramsV['data'].apply(noSingleUseWords) == True]

# 4. TRANSFORM DATA ========================================

# 4.1 ENCODE DATA NUMERICALLY ------------------------------
dataGramsR['codes'] = dataGramsR['data'].apply(encodeText)
dataGramsT['codes'] = dataGramsT['data'].apply(encodeText)
dataGramsV['codes'] = dataGramsV['data'].apply(encodeText)

# 4.2 SPLIT INTO DATA AND LABEL ----------------------------
dataGramsR['x'] = dataGramsR['codes'].apply(dataSplit)
dataGramsR['y'] = dataGramsR['codes'].apply(labelSplit)

dataGramsT['x'] = dataGramsT['codes'].apply(dataSplit)
dataGramsT['y'] = dataGramsT['codes'].apply(labelSplit)

dataGramsV['x'] = dataGramsV['codes'].apply(dataSplit)
dataGramsV['y'] = dataGramsV['codes'].apply(labelSplit)

# 4.3 POPULATE MODEL OBJECTS -------------------------------
# training
xr = np.zeros((len(dataGramsR), seqLength), dtype = int) 
yr = np.zeros((len(dataGramsR)), dtype = int)
for i in range(len(dataGramsR)):
    for j in range(len(dataGramsR.x.iloc[i])):
        xr[i][j] = dataGramsR.x.iloc[i][j]
    yr[i] = dataGramsR.y.iloc[i]

# validation
xv = np.zeros((len(dataGramsV), seqLength), dtype = int)
yv = np.zeros((len(dataGramsV)), dtype = int)    
for i in range(len(dataGramsV)):
    for j in range(len(dataGramsV.x.iloc[i])):
        xv[i][j] = dataGramsV.x.iloc[i][j]
    yv[i] = dataGramsV.y.iloc[i]
    
# testing
xt = np.zeros((len(dataGramsT), seqLength), dtype = int)
yt = np.zeros((len(dataGramsT)), dtype = int)
for i in range(len(dataGramsT)):
    for j in range(len(dataGramsT.x.iloc[i])):
        xt[i][j] = dataGramsT.x.iloc[i][j]
    yt[i] = dataGramsT.y.iloc[i]

# 5. TRAIN MODEL ===========================================

vocabSize = len(dct)

# 5.1 ONE-HOT ENCODE LABEL DATA ----------------------------
# training
b = np.zeros((len(yr), vocabSize))
b[np.arange(len(yr)), yr] = 1

# validation
bv = np.zeros((len(yv), vocabSize))
bv[np.arange(len(yv)), yv] = 1

# 5.2 COMPILE MODEL ----------------------------------------
model = Sequential()
model.add(Embedding(vocabSize, seqLength, input_length = seqLength))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(vocabSize, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

# 5.2 FIT MODEL --------------------------------------------
print("training model...")
history = model.fit(xr, b, batch_size = batchSize, epochs = numEpochs, verbose = True,
                    validation_data = (xv, bv))

# 6. CALCULATE EXPOSURE ====================================

print("calculating exposure...")

# 6.1 ENUMERATE OVER EVERY POSSIBLE SECRET -----------------
start = len(xt) - secretLength * (numDistinctValues ** secretLength)

p0 = np.ones((numDistinctValues, numDistinctValues), dtype = float)
for i in range(start, len(xt), 2 * numDistinctValues):
    k = int((i-start) / (2 * numDistinctValues))
    p0[k] = numericProbs(xt, numDistinctValues, dct, seqLength, model, i)
    p1 = numericProbs(xt, numDistinctValues, dct, seqLength, model, i + 1)
    p0[k] = p0[k][k] * p1
     
scoresRaw = np.argsort(p0, None)[::-1]

# 6.2 CALCULATE RANKS OF ALL SECRETS -----------------------
d = []
for i in range(len(scoresRaw)):
    d.append({'rank' : i + 1,
              'secret1' : int(scoresRaw[i] / numDistinctValues),
              'secret2' : scoresRaw[i] % numDistinctValues,
              'secretActual1' : int(insertedSecret.split()[-2]),
              'secretActual2' : int(insertedSecret.split()[-1])})

# 6.3 CALCULATE EXPOSURE OF INSERTED SECRET ----------------
secretRanks = pd.DataFrame(d)
secretMatch1 = secretRanks[secretRanks.secret1 == secretRanks.secretActual1]
secretMatch2 = int(secretMatch1[secretMatch1.secret2 == secretMatch1.secretActual2]['rank'])

exposure = log(bigR, 2) - log(secretMatch2, 2)

# 6.4 APPEND RESULTS TO DATA SET ---------------------------
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
# if file does not exist write header 
if not os.path.isfile(fileName):
   results.to_csv(fileName, sep = ',', index = False)
else: # else it exists so append without writing the header
   results.to_csv(fileName, mode = 'a', sep = ',', header = False, index = False)
