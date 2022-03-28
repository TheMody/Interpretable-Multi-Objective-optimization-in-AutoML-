
import torch.nn as nn
import numpy as np
import random
import string
import torch


#input is nlp model with .embed and .classify, val data, metric with input y,ypred and out score
def robustness_nlp_model(acc, model, data, metric):
    drop = nn.Dropout(p=0.4)
    def prediction(X):
        emb = model.embed(X)
        X = drop(emb)
        y = model.classify(X)
        return y
    overall = 0
    X, y = data
    y_pred = prediction(X)
    if not (type(y_pred) == np.ndarray):
        y_pred = np.argmax(y_pred.to("cpu").detach().numpy(), axis=1)
    score = metric(y, y_pred)
    return 1 +  (np.mean(score)- acc) / acc


def noise_func(X):
    noise_X = []
    randomnoisecount = 20
    for sample in X:
        if type(sample) == str: 
            for i in range(np.random.randint(randomnoisecount)):
                noise_place = np.random.randint(len(sample))
                sample = list(sample)
                sample[noise_place] = random.choice(string.ascii_letters)
                sample = "".join(sample)
            noise_X.append(sample)
        else:
            sentences = []
            for sentence in sample:
                for i in range(np.random.randint(randomnoisecount)):
                    noise_place = np.random.randint(len(sentence))
                    sentence = list(sentence)
                    sentence[noise_place] = random.choice(string.ascii_letters)
                    sentence = "".join(sample)
                sentences.append(sentence)
            noise_X.append((sentences[0], sentences[1]))
    return noise_X


def robustness_noise(y_pred,model, data, metric):
    X, y = data
    X_noise = noise_func(X)
    y_pred_noise = model.predict(X_noise)
    if model.type == 'nn':
       y_pred_noise = np.argmax(y_pred_noise.to("cpu").detach().numpy(), axis=1)
    score = metric(y, y_pred)
    scorenoise = metric(y, y_pred_noise)
    return 1 + (scorenoise-score) / score

def robustness_data_shift(y_pred,model, data, metric):
    X, y = data
    score = metric(y, y_pred)
    ids = []
    max_id = -1
    max_id_num = 0
    for a,unique in enumerate(np.unique(y)):
        id =[]
        for i,point in enumerate(y):
            if point == unique:
                id.append(i)
        ids.append(id)
        if len(id)> max_id_num:
            max_id = a
    ids[max_id] = ids[max_id][:round(len(ids[max_id])*0.2)]
    new_ids = []
    for id in ids:
        for p in id:
            new_ids.append(p)
    y = y[new_ids]
    y_pred = y_pred[new_ids]
    score_shift = metric(y, y_pred)
    return 1 + (score_shift-score) / score

