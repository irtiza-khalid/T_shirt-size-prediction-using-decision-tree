import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle

height = [158,158,160,160,165,165]
weight = [58,63,59,64,62,65]
t_shirt_size = ["Medium","Medium","Medium", "Large", "Large", "Large"]


le = preprocessing.LabelEncoder()
label = le.fit_transform(t_shirt_size)


features = list(zip(height, weight))

mymodel = KNeighborsClassifier(n_neighbors=3)
mymodel.fit(features, label)
model = pickle.load(open(r"C:\Users\SWishHy\Desktop\flask projects\AI_1\mymodel_knn.pkl", "rb"))

print("Sucess")
