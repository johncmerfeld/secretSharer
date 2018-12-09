import numpy as np
import re
from random import randint

def displayNumericResults(c, p, d, i):
    result = np.zeros((c, 2), dtype = float)
    for j in range(c):
        a = comboString(j)
            
        result[j][0] = j
        result[j][1] = p[i][d[a]]
        
    return (result[result[:, 1].argsort()])[::-1]

def cleanSMS(sms):
    
    # leetspeak
    sms = re.sub("[\.,]", " ", sms)
    sms = re.sub(" {2,}", " ", sms)
    sms = re.sub(" 2 ", " to ", sms)
    sms = re.sub(" 4 | fr ", " for ", sms)
    
    sms = re.sub(" abt ", " about ", sms)
    sms = re.sub(" aft ", " after ", sms)
    sms = re.sub(" ard ", " around ", sms)
    
    sms = re.sub(" ar ", " all right ", sms)
    sms = re.sub(" ar$", " all right", sms)
    
    sms = re.sub(" b ", " be ", sms)
    sms = re.sub(" bcz ", " because ", sms)
    sms = re.sub(" bday ", " birthday", sms)
    sms = re.sub(" brin ", " bring ", sms)
    
    sms = re.sub(" btw ", " by the way ", sms)
    sms = re.sub(" btw$", " by the way", sms)
    
    sms = re.sub(" buk ", " book ", sms)
    
    sms = re.sub(" c ", " see ", sms)
    sms = re.sub("^c ", "see ", sms)
    
    sms = re.sub(" coz | cuz | cos ", " cause ", sms)
    sms = re.sub("^coz |^cuz |^cos ", "cause ", sms)
    
    sms = re.sub(" da ", " the ", sms)
    sms = re.sub(" dat ", " that ", sms)
    
    sms = re.sub(" den ", " then ", sms)
    sms = re.sub("^den ", "then ", sms)
    sms = re.sub(" den$", " then", sms)
    
    sms = re.sub(" dint? ", " did not ", sms)
    
    sms = re.sub(" dis ", " this ", sms)
    sms = re.sub(" dis$", " this", sms)
    
    sms = re.sub(" dem | dm ", " them ", sms)
    sms = re.sub(" dey ", " they ", sms)
    sms = re.sub("^dey ", "they ", sms)
    sms = re.sub(" dnt ", " do not ", sms)
    
    sms = re.sub(" dun | don ", " do not ", sms)
    sms = re.sub("^dun |^don ", "do not ", sms)
    sms = re.sub(" dun$| don$", " do not", sms)
    
    sms = re.sub(" e ", " the ", sms)
    sms = re.sub(" esp " , " especially ", sms)
    sms = re.sub(" enuff ", " enough ", sms)
    sms = re.sub(" frens ", " friends ", sms)
    
    sms = re.sub(" fren " , " friend ", sms)
    sms = re.sub(" fren$", " fren", sms)
    
    sms = re.sub(" frm ", " from ", sms)
    
    sms = re.sub(" gd ", " good ", sms)
    sms = re.sub("^gd ", "good ", sms)
    sms = re.sub(" gd$", " good", sms)
    
    sms = re.sub(" gn ", " good night ", sms)
    sms = re.sub("^gn ", "good night ", sms)
    sms = re.sub(" gn$", " good night", sms)
    
    sms = re.sub("^hai ", "hey ", sms)
    
    sms = re.sub(" haf | hv | hav ", " have ", sms)
    sms = re.sub(" haf$| hv$| hav$", " have", sms)
    
    sms = re.sub(" haven ", " have not ", sms)
    
    sms = re.sub(" hse ", " house ", sms)
    sms = re.sub(" hse$", " house", sms)
    sms = re.sub(" hw ", " homework ", sms)
    sms = re.sub("^hw ", "how ", sms)
    
    sms = re.sub(" i ll ", " i will ", sms)
    sms = re.sub("^i ll ", "i will ", sms)
    sms = re.sub(" i ve ", " i have ", sms)
    sms = re.sub("^i ve ", "i have ", sms)
    
    sms = re.sub(" juz | jus | jos ", " just ", sms)
    sms = re.sub("^juz |^jus |^jos ", "just ", sms)
    
    sms = re.sub("kd ", "ked ", sms)
    sms = re.sub(" knw ", " know ", sms)
    
    sms = re.sub(" lar | lter ", " later ", sms)
    sms = re.sub(" lar$| lter$", " later", sms)
    sms = re.sub("^lar |^lter ", "later ", sms)
    
    sms = re.sub(" lib ", " library ", sms)
    sms = re.sub(" lib$", " library", sms)
    
    sms = re.sub(" lect ", " lecture ", sms)
    sms = re.sub("^ll ", "i will ", sms)
    sms = re.sub(" lyk ", " like ", sms)
    sms = re.sub(" m ", " am ", sms)
    sms = re.sub("^m ", "i am ", sms)
    sms = re.sub(" mayb ", " maybe ", sms)
    sms = re.sub(" meh ", " me ", sms)
    sms = re.sub(" msg ", " message ", sms)
    sms = re.sub(" neva ", " never ", sms)
    sms = re.sub(" mum ", " mom ", sms)
    sms = re.sub(" muz ", " must ", sms)
    sms = re.sub(" n ", " and ", sms)
    sms = re.sub("nd ", "ned ", sms)
    sms = re.sub(" nite ", " night ", sms)
    sms = re.sub(" noe ", " know ", sms)
    
    sms = re.sub(" nt ", " not ", sms)
    sms = re.sub("^nt ", "not ", sms)
    
    sms = re.sub(" nvm ", " never mind ", sms)
    sms = re.sub(" nvr ", " never ", sms)
    sms = re.sub(" nw ", " now ", sms)
    
    sms = re.sub(" nxt ", " next ", sms)
    sms = re.sub("^nxt ", "next ", sms)
    
    sms = re.sub(" okie | ok | k ", " okay ", sms)
    sms = re.sub("^okie |^ok |^k ", "okay ", sms)
    sms = re.sub(" okie$| ok$| k$", " okay", sms)
    
    sms = re.sub(" oredi | alr ", " already ", sms)
    sms = re.sub(" oredi$| alr$", " already", sms)
    
    sms = re.sub(" oso ", " also ", sms)
    
    sms = re.sub(" plz ", " please ", sms)
    sms = re.sub("^plz ", "please ", sms)
    sms = re.sub(" plz$", " please", sms)
    
    sms = re.sub(" pple? ", " people ", sms)
    
    sms = re.sub(" pg ", " page ", sms)
    sms = re.sub(" pg$", " page", sms)
    
    sms = re.sub(" r ", " are ", sms)
    sms = re.sub("^r ", "are ", sms)
    sms = re.sub(" r$", " are", sms)
    
    sms = re.sub(" rem ", " remember ", sms)
    sms = re.sub(" rite ", " right ", sms)
    
    sms = re.sub(" rly ", " really ", sms)
    sms = re.sub("^rly ", "really ", sms)
    sms = re.sub(" rly$", " really", sms)
    
    sms = re.sub(" ru ", " are you ", sms)
    sms = re.sub(" s ", " is ", sms)
    sms = re.sub("^s ", "its ", sms)
    
    sms = re.sub(" sch ", " school ", sms)
    sms = re.sub(" sch$", " school", sms)
    
    sms = re.sub(" shd | shld ", " should ", sms)
    sms = re.sub(" slp ", " sleep ", sms)
    
    sms = re.sub(" sme", " some", sms)
    sms = re.sub("^sme", "some", sms)
    
    sms = re.sub(" smth ", " something ", sms)
    
    sms = re.sub(" tat ", " that ", sms)
    sms = re.sub("^tat ", "that ", sms)
    sms = re.sub(" tat$", " that", sms)
    
    sms = re.sub(" tmr | tml ", " tomorrow ", sms)
    sms = re.sub("^tmr |^tml ", "tomorrow ", sms)
    sms = re.sub(" tmr$| tml$", " tomorrow", sms)
    
    sms = re.sub(" thanx ", " thanks ", sms)
    sms = re.sub(" thanx$", " thanks", sms)
    sms = re.sub("^thanx ", "thanks ", sms)
    
    sms = re.sub(" thgt ", " thought ", sms)
    sms = re.sub(" thk | thnk ", " think ", sms)
    sms = re.sub(" tis ", " this ", sms)
    sms = re.sub(" tot " , " thought ", sms)
    sms = re.sub(" ttyl$", " talk to you later", sms)
    
    sms = re.sub(" tym ", " time ", sms)
    sms = re.sub(" tym", " time", sms)
    
    sms = re.sub(" [uüü] ", " you ", sms)
    sms = re.sub("^[uüü] ", "you ", sms)
    sms = re.sub(" [uüü]$", " you", sms)
    
    sms = re.sub(" ur ", " your ", sms)
    sms = re.sub(" v ", " very ", sms)
    sms = re.sub(" vil ", " will ", sms)
    sms = re.sub("^ve ", "i have ", sms)
    sms = re.sub(" wan ", " want ", sms)
    sms = re.sub(" w ", " with ", sms)
    
    sms = re.sub(" wana ", " wanna ", sms)
    sms = re.sub("^wana ", "wanna ", sms)
    
    sms = re.sub(" wat ", " what ", sms)
    sms = re.sub("^wat ", "what ", sms)
    sms = re.sub(" wat$", " what", sms)
    
    sms = re.sub(" wen ", " when ", sms)
    sms = re.sub("^wen ", "when ", sms)
    
    sms = re.sub(" wif | wid | wth ", " with ", sms)
    sms = re.sub("^wif |^wid |^wth ", "with ", sms)
    sms = re.sub(" wif$| wid$| wth$", " with", sms)
    
    sms = re.sub(" wk ", " week ", sms)

    sms = re.sub(" wun ", " wont ", sms)
    
    sms = re.sub(" y ", " why ", sms)
    sms = re.sub("^y ", "why ", sms)
    sms = re.sub(" y$", " why", sms)
    
    sms = re.sub("yup", "yep", sms)

    # remove laughter and smiles
    sms = re.sub(" d ", " ", sms)
    sms = re.sub(" d$", "", sms)
    sms = re.sub("^d ", "", sms)
    sms = re.sub(" ha ", " ", sms)
    sms = re.sub("^ha ", "", sms)
    sms = re.sub(" ha$, ", "", sms)
    sms = re.sub(" lor ", " ", sms)
    sms = re.sub(" lor$", "", sms)
    sms = re.sub(" lols? ", " ", sms)
    sms = re.sub("^lols? ", "", sms)
    sms = re.sub(" lols?$", "", sms)
    sms = re.sub("a*(ha){2,}h*", "", sms)
    sms = re.sub(" hee ", " ", sms)
    sms = re.sub("^hee ", "", sms)
    sms = re.sub(" hee$", "", sms)
    
    # remove words I don't understand
    sms = re.sub(" lei ", " ", sms)
    sms = re.sub("^lei ", " ", sms)
    sms = re.sub(" lei$", " ", sms)
    
    # standardize most '-ing' to '-in'
    sms = re.sub("(?<=[bdfghklmnoprstvwy])ing(?= )", "in", sms)
    sms = re.sub("(?<=[bdfghklmnoprstvwy])ing$", "in", sms)
    
    # force spaces between comma- or period-separated words
    sms = re.sub("(?<=[^ ])[\.,](?=[^ ])", " ", sms)
    
    return sms

def dataSplit(tup):
    n = len(tup)
    return tup[0 : (n - 1)]

def labelSplit(tup):
    n = len(tup)
    return tup[n - 1]

def learnSecret():
   
    # for each digit d in s
    #     for each number i in R
    #         get Pr(model.predict('my locker combination is [i]'))
    #     
    
    return None

# get word from dictionary ID
def getWord(d, i):
    return list(d.keys())[list(d.values()).index(i)]
    
# see prediction vs actual sentence
def showResult(x, ya, yp, d):
    s = ""
    for i in range(len(x)):
        s += getWord(d, x[i]) + " "
    
    s1 = s + " " + getWord(d, yp)
    s2 = s + " " + getWord(d, ya)
    
    print("Actual: ", s2, "\nPredicted: ", s1, "\n")

def showResults(x, ya, yp, i, d):
    showResult(x[i], ya[i], yp[i], d)
 
# see other predicted words
def showOptions(x, ya, yp, n, d, p, i):
    showResult(x[i], ya[i], yp[i], d)
    print("Prediction ideas:")
    
    ps = -np.sort(-p[i])
    pa = np.abs(-np.argsort(-p[i]))
    
    for j in range(n):
        print(j + 1, ". ", getWord(d, pa[j]), " (", round(ps[j] * 100, 2), "%)", sep = '')

def generateSecret(length, size):
    secret = ""
    for i in range(length):
        a = randint(0, size)
        if a < 10:
            a = "0" + str(a)
        a = str(a)
        secret = secret + a + " "
    
    return secret[:-1]

def discoverSecret(x, m, gs, i, sl):
    
    secret = ""
    
    xn = np.zeros((sl, gs), dtype = float)
    for j in range(sl):
        for k in range(gs):
            xn[j][k] = x[i][k]

    p0 = m.predict_classes(xn)
    
    for j in range(sl):
        secret += str(p0[0]) + " "
        for j in range(sl):
            for k in range(gs - 1):
                xn[j][k] = xn[j][k + 1]
          
        xn[:, gs -1] = p0
        
        p0 = m.predict_classes(xn)
 
    return secret

def comboString(i):
    return str(i)
    

def enumerateSecrets(length, size, rid, pref):
    d = []
    
    if length == 1:
        for i in range(size):
            a = pref + comboString(i)
            d.append({'id' : rid,
                      'text' : a,
                      'noPunc' : a,
                      'splchk' : a})
            rid += 1
    
    if length == 2:
        for i in range(size):
            a = pref + comboString(i)
            for j in range(size):
                b = a + " " + comboString(j)
                d.append({'id' : rid,
                          'text' : b,
                          'noPunc' : b,
                          'splchk' : b})
                rid += 1
                
    if length == 3:
        for i in range(size):
            a = pref + comboString(i)
            for j in range(size):
                b = a + " " + comboString(j)
                for k in range(size):
                    c = b + " " + comboString(k)
                    d.append({'id' : rid,
                              'text' : c,
                              'noPunc' : c,
                              'splchk' : c})
                    rid += 3             
    
    if length == 4:
        for i in range(size):
            a = pref + comboString(i)
            for j in range(size):
                b = a + " " + comboString(j)
                for k in range(size):
                    c = b + " " + comboString(k)
                    for q in range(size):
                        d = c + " " + comboString(q)
                        d.append({'id' : rid,
                                  'text' : d,
                                  'noPunc' : d,
                                  'splchk' : d})
                        rid += 1                    
    return d, rid

def numericProbs(x, size, d, gs, m, i ): 
    xn = np.zeros((1, gs), dtype = float)
    for k in range(gs):
        xn[0][k] = x[i][k]

    p0 = m.predict(xn)[0]
    
    numericProbs = np.zeros((size), dtype = float)
    
    for j in range(size):
        a = comboString(j)
        numericProbs[j] = p0[d[a]]
        
    return numericProbs