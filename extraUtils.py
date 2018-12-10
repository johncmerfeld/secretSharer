import numpy as np
from random import randint

def displayNumericResults(c, p, d, i):
    result = np.zeros((c, 2), dtype = float)
    for j in range(c):
        a = str(j)
            
        result[j][0] = j
        result[j][1] = p[i][d[a]]
        
    return (result[result[:, 1].argsort()])[::-1]

    
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
