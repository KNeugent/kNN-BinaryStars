# kNN-BinaryStars

This program differentiates between single and binary star systems using photometric constraints and a k-NN classifier. It was originally developed as part of my graduate thesis work at the University of Washington to differentiate between red supergiants with and without OB star companions, but it can be extended to any set of binary systems where the photometric colors of the two stars are sufficiently different (as in, one is hot / blue and one is cold / red).

## Methodology

As is shown in the image below, in the visible wavelengths, red supergiants are red, and OB stars are blue. If a red supergiant has an OB companion, it will therefore appear slightly more yellow than a single red supergiant.
![RSGOB](https://github.com/KNeugent/kNN-BinaryStars/blob/main/images/redBlue.jpg)

As part of my thesis work, I trained a [k-Nearest Neighbor (k-NN) classifier using python's sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) to separate out the single red supergiants from the binary red supergiants based on this difference in color. In this simple example below, the k-NN classifier is operating in two dimensions (along the x and y-axis). It knows about two different classes of objects and wants to figure out which class the new object falls into. It does this by placing the new object in it's position in 2D space and then figuring out if it falls closer to the objects of class A or B.
![knn1](https://github.com/KNeugent/kNN-BinaryStars/blob/main/images/knn1.jpg)

In the case of the red supergiant and OB star binaries, the axes were photometric color information spanning from the UV to the NIR (in 7-dimentional space) and the two classes were single red supergiants and binary red supergiants.
![knn2](https://github.com/KNeugent/kNN-BinaryStars/blob/main/images/knn2.jpg)

To train the k-NN classifier, I heavily relied on the following [blog post](https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a) and will go into a bit more detail below.

### Training and Applying the k-NN classifier

The k-NN model was trained on the 295 red supergiants (single and binary) that had been observed spectroscopically and were thus known to be either single or binary. The features used were the following:
* Magnitudes: Umag, Bmag, Vmag, Imag
* Colors: U-B, B-V
* Flag: UV (0 if no detection in UV, 1 if detection in UV)

The first step was to scale the feature values using `sklearn RobustScaler`. This is because the magnitude values ranged from 15 - 20, the colors ranged from -5 to 5, and the UV flag was either 0 or 1. Instead of having the magnitudes get more weight because their values were larger, I wanted each of the features to get the same weight when classifying. Scaling the features achieves this.

The next step was to define the classifier using the spectroscopically confirmed / known single and binary red supergiants. Using the method described in the [blog post](https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a), I determined that the optimal number of folds for cross-validation was 15 (`cv = 15`). I then iterated over the `leaf_size`, `n_neighbors`, and `p` value to find the best set of hyperparameters for classification by maximizing the model's overall score.

For reference, I found the following results were ideal for my set of data:
* cv = 15; folds for cross-validation / how many times is the test dataset (the knowns) split up and tested. In this case, it will be split up into 15 groups and tested 15 different times. Since there are 295 supergiants, this results in around 20 stars for each test.
* leaf_size = 1; changes the speed of the construction and query. Since my dataset is relatively small, this value isn't super important. 
* p = 1; determines how the distance between stars is determined. The p is the p value in the [Minkowski Distance](https://python-course.eu/machine-learning/k-nearest-neighbor-classifier-in-python.php), and a p=1 effectively means that a manhattan distance is used instead of a euclidian distance. Changing the p value for this set of features makes very little difference. 
* n_neighbors = 26; number of neighboring stars included in the calculation. Since my starts are relatively clustered in feature-space, it is good to have this value a little on the higher side (the default is 5). 
* best score = 0.9339; This is essentially the error and can be estimated using the set of known stars. 

Finally, I applied this k-NN model to the 3698 unclassified red supergiants and produced the results below.

### Results

After running the k-NN classifier on the full sample of objects, I was able to assign a percentage chance of binarity to each individual star based on their photometric colors. The outcome is shown in the figure below.
![knnLMC](https://github.com/KNeugent/kNN-BinaryStars/blob/main/images/LMCknn.jpg)

As expected, stars with excess blue light are more likely to be binary red supergiants with OB stars companions and those that don't have any blue light are more likely to be single red supergiants.

By summing up the probability that each star is either a likely binary or single star and taking the error in the k-NN classification into account, I can then determine an overall binary fraction. The errors on the binary fraction are then calculated by first assuming that all of the error in the k-NN classification overestimates the number of binaries, and then by assuming that it always underestimates the number of binaries. This is why the errors are not symmetrical. The final result for the test dataset is `10% +18.7 -3.6`.

## Using this code

### Dependencies

The imported packages are pandas, numpy, and sklearn.

This code has been tested using python 3.7.3, numpy 1.18.2, pandas 1.0.3, and sklearn 0.21.

### Running the code

At the most basic level, the following must be changed in the main method to apply a k-NN classification algorithm to a new set of objects:
* update the `knownStars` and `unknownStars` arrays. The `knownStars` must have a column called "Classification" that contains a flag indicating binarity (0 = single star, 1 = binary).
* update the `features` array to select the relevant features to train the model against.

However, this k-NN model has been optimized for my set of data and will need to be optimized for yours as well. Please follow the blog post mentioned above and examine the comments in the code to build your own k-NN model and classify new stars.

### Outputted files

Currently, no files are outputted. However, there are several print statements in the main method that output the best k-NN parameters as well as the final binary fraction with errors, as is shown below.

`Best leaf_size: 1
Best p: 1
Best n_neighbors: 26
Best score: 0.93
The binary fraction for this sample is: 10.63 plus 18.67 minus 3.61`