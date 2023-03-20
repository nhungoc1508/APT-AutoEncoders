import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold,
)

def get_loss_fl(model, data):
    rec = model.predict(data)
    loss = tf.reduce_sum(tf.math.abs(tf.cast(data, tf.float32) - rec), axis=1).numpy()
    return rec, loss

def discounted_cumulative_gain(ranks):
    dcg = 0.0
    for rank in ranks:
        dcg = dcg + 1.0/np.log2(rank+1)
    return dcg

# Calculate max possible DCG and ratio
def normalized_discounted_cumulative_gain(ranks,num_gt):
    dcg = discounted_cumulative_gain(ranks)
    maxdcg = 0.0
    for i in range(1,num_gt+1):
        maxdcg = maxdcg + 1.0/np.log2(i+1)
    return (dcg/maxdcg)

def list_to_txt(arr, file):
    for i in arr:
        file.write(str(i) + "\n")
    file.close()