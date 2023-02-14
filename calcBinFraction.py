import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def scaleData(known, unknown, features):
    scale = RobustScaler()
    for feature in features:
        knownScaled[feature] = scaler.fit_transform(known[feature])
        unknownScaled[feature] = scaler.fit_transform(unknown[feature])

    return knownScaled, unknownScaled

def defineClassifier(known):
    X = known
    y = known['Classification'].values

    knn = KNeighborsClassifier(weights="distance")

    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p = [1,2]
    hyperparmeters = dict(leaf_size=leaf_size,n_neighbors=n_neighbors,p=p)

    cv = 15
    knn_gscv = GridSearchCV(knn,hyperparameters,cv=cv)
    knn_gscv.fit(X,y)

    bestLeaf = knn_gscv.best_estimator_.get_params()['leaf_size']
    bestP = knn_gscv.best_estimator_.get_params()['p']
    bestNNeigh = knn_gscv.best_estimator_.get_params()['n_neighbors']
    bestScore = knn_gscv.best_score_

    return bestLeaf, bestP, bestNNeigh, bestScore

def classifyStars(unknownStars, leaf, pVal, nneigh):
    knn = KNeighborsClassifier(weights="distance")
    hyperparmeters = dict(leaf_size=leaf,n_neighbors=nneigh,p=pVal)
    cv = 15
    knn_gscv = GridSearchCV(knn,hyperparameters,cv=cv)
    predProb = knn_gscv.predict_proba(unknownStars)

    return predProb
    
def calcBinFraction(knownStars, unknownStars, features):
    knownScaled, unknownScaled = scaleData(knownStars, unknownStars, features)
    bestLeaf, bestP, bestNNeigh, bestScore = defineClassifier(knownStars)
    predProb = classifyStars(unknownStars)

    binG = np.sum(predProb[:,1])
    binK = len(known[known["Classification"]==1])

    singG = np.sum(predProb[:,0])
    singK = len(known[known["Classification"]==0])

    errorV = len(predProb)-len(predProb)*0.9338596491228071

    per = (binG+binK)/(singG+singK)*100
    perM = (binG-errorV+binK)/(singG+errorV+singK)*100
    perP = (binG+errorV+binK)/(singG-errorV+singK)*100

def main():
    # csv of known binary or single stars as well as their photometry / flags
    # column named "classification" with 0 or 1 for binary or non-binary
    knownStars = pd.read_csv("KnownStars.csv")

    # csv of unknown stars as well as their photometry / flags
    unknownStars = pd.read_csv("UnknownStars.csv")

    # features
    features = [['Umag','Bmag','Vmag','Imag','U-B','B-V','UV']]

if __name__ == "__main__":
    main()
