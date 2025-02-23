import random
import numpy as np

def traintest(x,y,size):
    l=list(enumerate(x))
    arr=np.arange(len(x))
    random.shuffle(arr)
    per=size*len(x)
    x_train,x_test,y_train,y_test=[],[],[],[]
    cou=0
    for i in arr:
        cou+=1
        if cou > per:
         x_train.append(l[i][1])
         y_train.append(y[l[i][0]])
         continue
        x_test.append(l[i][1])
        y_test.append(y[l[i][0]])
    return np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)