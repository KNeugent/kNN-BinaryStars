import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def scale_data(known, unknown, features):
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


def define_classifier(known, unknown):
    """
    Iterates over a range of hyperparameters and defines a k-NN classifier.
    Then uses this classifier to classify the unknown stars.

        Parameters:
            known (pandas dataFrame): Pandas DF with scaled features
            unknown (pandas dataFrame): Pandas DF with scaled features

        Returns:
            best_leaf (int): best leaf_size value
            best_p (int): best p-value
            best_NNeigh (int): best n_neighbor value
            best_score (float): best score for k-NN classifier
            pred_prob ([float]): array of output values from k-NN
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
    best_leaf = knn_gscv.best_estimator_.get_params()["leaf_size"]
    best_p = knn_gscv.best_estimator_.get_params()["p"]
    best_NNeigh = knn_gscv.best_estimator_.get_params()["n_neighbors"]
    best_score = knn_gscv.best_score_

    # classify the remaining unknown stars
    pred_prob = knn_gscv.predict_proba(unknown)

    return best_leaf, best_p, best_NNeigh, best_score, pred_prob


def calc_bin_fraction(pred_prob, known, best_score):
    """
    Calculates the binary fraction with errors based on the result from
    the k-NN classifier

        Parameters:
            pred_prob ([float]): array of output values from k-NN
            known (pandas DataFrame): pandas DataFrame of known stars
            best_score (float): best score (for calculating the error)

        Returns:
            per (float): binary fraction as a percent
            err_per_M (float): negative error on binary fraction
            err_per_P (float): positive error on binary fraction
    """
    # select the binaries from known and unknown
    bin_g = np.sum(pred_prob[:, 1])
    bin_k = len(known[known["Classification"] == 1])

    # select the single stars from known and unknown
    sing_g = np.sum(pred_prob[:, 0])
    sing_k = len(known[known["Classification"] == 0])

    # calculate the error (only on the unknown stars)
    error_v = len(pred_prob) - len(pred_prob) * best_score

    # determine the high and low values for the error
    # note that the error only needs to be applied to the binaries
    # see discussion in README for rational behind this calculation
    per = (bin_g + bin_k) / (sing_g + sing_k) * 100
    err_per_M = (bin_g - error_v + bin_k) / (sing_g + error_v + sing_k) * 100
    err_per_P = (bin_g + error_v + bin_k) / (sing_g - error_v + sing_k) * 100

    return per, err_per_M, err_per_P


def main():
    # csv of known binary or single stars as well as their photometry / flags
    # column named "classification" with 0 or 1 for binary or non-binary
    known_stars = pd.read_csv("known_stars.csv")

    # csv of unknown stars as well as their photometry / flags
    unknown_stars = pd.read_csv("unknown_stars.csv")

    # features
    features = [["Umag", "Bmag", "Vmag", "Imag", "U-B", "B-V", "UV"]]

    # scale the features so they are weighted equally
    known_scaled, unknown_scaled = scale_data(known_stars, unknown_stars, features)

    # train the classifier using the known stars and classify unknown stars
    best_leaf, best_p, best_NNeigh, best_score, pred_prob = define_classifier(
        known_stars, unknown_stars
    )

    # print the best values
    print("Best leaf_size:", best_leaf)
    print("Best p:", best_p)
    print("Best n_neighbors:", best_NNeigh)
    print("Best score:", round(best_score, 2))

    # calculate the binary fraction
    per, err_per_M, err_per_P = calc_bin_fraction(pred_prob, known_stars, best_score)

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
