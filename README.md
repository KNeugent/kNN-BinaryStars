# kNN-BinaryStars

This program differentiates between single and binary star systems using photometric constraints and a k-NN classifier. It was originally developed as part of my graduate thesis work at the University of Washington to differentiate between red supergiants with and without OB star companions, but it can be extended to any set of binary systems where the photometric colors of the two stars are sufficiently different (as in, one is hot / blue and one is cold / red).

## Methodology

As is shown in the image below, in the visible wavelengths, red supergiants are red, and OB stars are blue. If a red supergiant has an OB companion, it will therefore appear slightly more yellow than a single red supergiant.
![RSGOB](https://github.com/KNeugent/kNN-BinaryStars/blob/main/images/redBlue.jpg)

As part of my thesis work, I trained a k-Nearest Neighbor (k-NN) classifier to separate out the single red supergiants from the binary red supergiants based on this difference in color. In this simple example below, the k-NN classifier is operating in two dimensions (along the x and y-axis). It knows about two different classes of objects and wants to figure out which class the new object falls into. It does this by placing the new object in it's position in 2D space and then figuring out if it falls closer to the objects of class A or B.
![knn1](https://github.com/KNeugent/kNN-BinaryStars/blob/main/images/knn1.jpg)

In the case of the red supergiant and OB star binaries, the axes were photometric color information spanning from the UV to the NIR and the two classes were single red supergiants and binary red supergiants.
![knn2](https://github.com/KNeugent/kNN-BinaryStars/blob/main/images/knn2.jpg)

To train the k-NN classifier, I heavily relied on the following [blog post](https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a) and will go into a bit more detail below.

### Training and Applying the k-NN classifier



### Results

After running the k-NN classifier on the full sample of objects, I was able to assign a percentage chance of binarity to each individual star based on their photometric colors. The outcome is shown in the figure below.
![knnLMC](https://github.com/KNeugent/kNN-BinaryStars/blob/main/images/LMCknn.jpg)

As expected, stars with excess blue light are more likely to be binary red supergiants with OB stars companions and those that don't have any blue light are more likely to be single red supergiants.

## Using this code

### Dependencies

The imported packages are pandas, numpy, and sklearn.

This code has been tested using python 3.7.3, numpy 1.18.2, pandas 1.0.3, and sklearn 0.21.

### Running the code

### Outputted files

