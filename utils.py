import time
import math
Big_C = 100
def Prediction_timed(model, X):
    start = time.time()
    fY = model.predict(X)
    end = time.time()
    return fY, (end-start)/len(X)

def time_sigmoid(x,wished_time):
    return -1.0/(1.0+ math.exp( (-x + wished_time)  / (wished_time/5.0))) +1
    
def score(x, w_x):
    return  ((Big_C * delta(x, w_x) + 1)* x + Big_C * (1-delta(x, w_x)) * w_x ) 

def score_scaled(x, w_x):
    return  score(x,w_x) * Big_C / score(1,w_x)
    
def sigmoid(x,w_x):
    return -1.0/(1.0+ math.exp( (-x + w_x)  / (w_x/5.0))) +1    

def delta(x, threshold):
    if x < threshold:
        return 1
    else:
        return 0
    
def deltareverse(x, threshold):
    if x < threshold:
        return 0
    else:
        return 1
    
def time_score(t, w_t):
   return (10/6)*(( Big_C * delta(t, w_t) + 1)* time_sigmoid(t, w_t) + Big_C * (1-delta(t, w_t)) * (1-time_sigmoid(w_t,w_t)) - Big_C *0.5)