from random import randint
import numpy as np
from numpy import array


iterations = 2  # Samples used to estimate the Rademacher correlation


def main():
    training_vector = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]

    rademacher_complexity = rademacher_estimate(training_vector, origin_plane_classifier)
    print("Rademacher correlation of origin centered plane classifier:", rademacher_complexity)


"""
===============================================================================================================

    - Rademacher Complexity Functions

===============================================================================================================
"""


def rademacher_estimate(training_vector, classifier):

    """
    Given a (training) data-set, estimate the upper genralisation bounds using emperical Rademacher Complexity.
    This is performed over a number of estimations taking the average, which will give close to the expected
    value due to the law of large numbers (LLN).

    :param training_vector: a vector of training data (pre-defined).
    :param classifier: a function that generates an iterator over hypotheses given a data-set
    """

    totals = list()  # List of upper bounds from each iteration.

    for i in range(iterations):  # Iterating multiple times to get a good approximation.

        # Generate a tuple of random_vector - random variables (+1, -1)
        random_vector = rademacher_random_variable(len(training_vector))

        # Finding the Rademacher correlations/bounds on the random_vector.
        bounds = [h.correlation(training_vector, random_vector) for h in list(classifier(training_vector))]

        # Adding the upper bound to the totals list.
        totals.append(max(bounds))

        print("==========================================================================")
        print("Bounds:", bounds)
        print("Upper Bounds:", max(bounds))
        print("Iteration:", i)
        print("==========================================================================")
        print()

    return sum(totals)/iterations  # Averaging all the upper bounds to come up with the best estimate.


def rademacher_random_variable(number):

    """
    Generate a desired number of rademacher_random_variable (with values {+1, -1})
    :param number: The number of Rademacher random variables to generate.
    """

    return [randint(0, 1) * 2 - 1 for x in range(number)]


"""
===============================================================================================================

    - Classifier Super Class

===============================================================================================================
"""


class Classifier:

    def correlation(self, training_vector, random_vector):

        """
        Return the rademacher correlation between a label assignment and the predictions of
        the classifier

        :param hypothesis_vector: A vector of the training data
        :param random_vector: A vector of Rademacher random variables (+1/-1)
        """

        print("Training Vector:", training_vector)
        hypothesis_vector = [1 if (self.classify(d)) else -1 for d in training_vector]

        # Product of random_vector
        dot_product = float(np.dot(hypothesis_vector, random_vector))

        # Number of features we are testing.
        size = float(len(training_vector))

        # Divide by size to push the number between 1 and -1.
        correlation = dot_product / size

        print("Training Vector:", training_vector)
        print("Rademacher Vector:", random_vector)
        print("Hypothesis Vector:", hypothesis_vector)
        print("Dot Product:", dot_product)
        print("Correlation:", correlation)
        print()

        return correlation


"""
===============================================================================================================

    - Origin Plane Classifier

===============================================================================================================
"""


class PlaneHypothesis(Classifier):

    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector
        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
               (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):

    """
    A class that represents a decision boundary that must pass through the
    origin.
    """

    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.
        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


def origin_plane_classifier(training_vector):

    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.
    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.
    Args:
      training_vector: The dataset to use to generate hypotheses
    """

    # Cloning the training data.
    copyDataset = np.array(training_vector)

    theta = np.multiply(np.arctan2(
        copyDataset[:, 1], copyDataset[:, 0]), 180 / np.pi)
    print("Old Theta:", theta)
    theta.sort()
    print("New Theta:", theta)

    hypothesesTheta = list()

    # Merging the array so [1,2,3,4] -> [1.5, 2.5, 3.5]
    for idx in range(len(theta) - 1):
        if (theta[idx] != theta[idx + 1]):
            meanTheta = (theta[idx] + theta[idx + 1]) / 2
            hypothesesTheta.append(meanTheta)

    hypothesesTheta.append(theta[-1] + np.spacing(np.single(1)))
    print("Hypo Theta:", hypothesesTheta)

    hypotheses = np.zeros((len(2 * hypothesesTheta), 2))
    print("Final Hypo:", hypotheses)

    idx1 = 0
    for idx2, theta in enumerate(hypothesesTheta, 0):
        print("Index 1: ", idx1)
        print("Index 2: ", idx2)


        hypotheses[idx1][0] = -1
        hypotheses[idx1][1] = np.tan(hypothesesTheta[idx2])

        hypotheses[idx1 + 1][0] = 1
        hypotheses[idx1 + 1][1] = -np.tan(hypothesesTheta[idx2])

        idx1 += 2

    print(hypotheses)
    for h in hypotheses:
        print("OPH", OriginPlaneHypothesis(h[0], h[1]))
        yield OriginPlaneHypothesis(h[0], h[1])


if __name__ == "__main__":
    main()
