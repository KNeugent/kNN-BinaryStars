import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def scaleData(known, unknown, features):
    """
    Scales the features numerically so ones with higher numerical values
    don't dominate the classification.

        Parameters:
            known (pandas dataFrame): Pandas DF of features for known stars
            unknown (pandas dataFrame): Pandas DF of features for unknown stars
            features (list): List of features to scale

        Returns:
            known (pandas dataFrame): Pandas DF with scaled features for known
            unknown (pandas dataFrame): Pandas DF with scaled features for unknown
    """
    # use the RobustScaler to scale the data
    scaler = RobustScaler()
    # scale the features for both the unknown and known stars
    for feature in features:
        known[feature] = scaler.fit_transform(known[feature])
        unknown[feature] = scaler.fit_transform(unknown[feature])

    return known, unknown


def defineClassifier(known, unknownStars):
    """
    Iterates over a range of hyperparameters and defines a k-NN classifier.
    Then uses this classifier to classify the unknown stars.

        Parameters:
            known (pandas dataFrame): Pandas DF with scaled features
            unknown (pandas dataFrame): Pandas DF with scaled features

        Returns:
            bestLeaf (int): best leaf_size value
            bestP (int): best p-value
            bestNNeigh (int): best n_neighbor value
            bestScore (int): best score for k-NN classifier
            predProb ([int,int,...]): array of output values from k-NN
    """

    # don't include Classification in training set since it isn't a feature
    X = known.drop(columns=["Classification"])
    # set the classification flag (0 = single star, 1 = binary)
    y = known["Classification"].values

    # set the weights to scale by distance so closer ones are valued higher
    knn = KNeighborsClassifier(weights="distance")

    # create a grid of hyperparameters to iterate over
    leaf_size = list(range(1, 50))
    # force a minimum number of neighbors to be 5 given the dataset size
    n_neighbors = list(range(5, 30))
    # the value of p isn't incredibly important for this particular use case
    p = [1, 2]
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    # set the optimal number of folds to 15 based on previous testing
    # note: see method discussed in README for more information
    # on how to test further
    cv = 15
    # iterate over the grid to find the best hyperparameters
    knn_gscv = GridSearchCV(knn, hyperparameters, cv=cv)
    # perform the fit with the trained k-NN model
    knn_gscv.fit(X, y)

    # save the results of the best set of parameters
    bestLeaf = knn_gscv.best_estimator_.get_params()["leaf_size"]
    bestP = knn_gscv.best_estimator_.get_params()["p"]
    bestNNeigh = knn_gscv.best_estimator_.get_params()["n_neighbors"]
    bestScore = knn_gscv.best_score_

    # classify the remaining unknown stars
    predProb = knn_gscv.predict_proba(unknownStars)

    return bestLeaf, bestP, bestNNeigh, bestScore, predProb


def calcBinFraction(predProb, known, bestScore):
    """
    Calculates the binary fraction with errors based on the result from
    the k-NN classifier

        Parameters:
            predProb ([int,int,...]): array of output values from k-NN
            known (pandas DataFrame): pandas DataFrame of known stars
            bestScore (int): best score (for calculating the error)

        Returns:
            per (int): binary fraction as a percent
            err_per_M (int): negative error on binary fraction
            err_per_P (int): positive error on binary fraction
    """
    # select the binaries from known and unknown
    binG = np.sum(predProb[:, 1])
    binK = len(known[known["Classification"] == 1])

    # select the single stars from known and unknown
    singG = np.sum(predProb[:, 0])
    singK = len(known[known["Classification"] == 0])

    # calculate the error (only on the unknown stars)
    errorV = len(predProb) - len(predProb) * bestScore

    # determine the high and low values for the error
    # note that the error only needs to be applied to the binaries
    # see discussion in README for rational behind this calculation
    per = (binG + binK) / (singG + singK) * 100
    err_per_M = (binG - errorV + binK) / (singG + errorV + singK) * 100
    err_per_P = (binG + errorV + binK) / (singG - errorV + singK) * 100

    return per, err_per_M, err_per_P


def main():
    # csv of known binary or single stars as well as their photometry / flags
    # column named "classification" with 0 or 1 for binary or non-binary
    knownStars = pd.read_csv("KnownStars.csv")

    # csv of unknown stars as well as their photometry / flags
    unknownStars = pd.read_csv("UnknownStars.csv")

    # features
    features = [["Umag", "Bmag", "Vmag", "Imag", "U-B", "B-V", "UV"]]

    # scale the features so they are weighted equally
    knownScaled, unknownScaled = scaleData(knownStars, unknownStars, features)

    # train the classifier using the known stars and classify unknown stars
    bestLeaf, bestP, bestNNeigh, bestScore, predProb = defineClassifier(knownStars, unknownStars)

    # print the best values
    print("Best leaf_size:", bestLeaf)
    print("Best p:", bestP)
    print("Best n_neighbors:", bestNNeigh)
    print("Best score:", round(bestScore, 2))

    # calculate the binary fraction
    per, err_per_M, err_per_P = calcBinFraction(predProb, knownStars, bestScore)

    # print the resulting binary fraction with positive and negative errors
    print(
        "The binary fraction for this sample is:",
        round(per, 2),
        "plus",
        round(err_per_P, 2),
        "minus",
        round(err_per_M, 2),
    )


if __name__ == "__main__":
    main()
