# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        #for each case, counts how many times a number came and what pixels were on for that pic 
       
        labelCounts = util.Counter()   
        labelTotals = util.Counter() 

        featureCounts = {}                        

        for label in self.legalLabels:
            featureCounts[label] = util.Counter()

        for i in range(len(trainingData)):
            datum = trainingData[i]             
            label = trainingLabels[i]

            labelCounts[label] += 1
            labelTotals[label] += 1

            for f, value in datum.items():
                if value > 0:                         
                    featureCounts[label][f] += 1

        # Probabilty 
        self.prior = util.Counter()
        totalExamples = len(trainingLabels)
        for label in self.legalLabels:
            #P( certain label) = count(label)/ total ex
            self.prior[label] = float(labelCounts[label]) / float(totalExamples)

        #Laplace smoothing and k
        #Small k = trust the data a lot
        #Big k = trust the data a little
        bestK = None
        bestAccuracy = -1.0
        bestConditionalProb = None

        for k in kgrid:
        # make conditional probabilities for this k
            conditionalProb = {}                   

            for label in self.legalLabels:
                cond = util.Counter()
                N_y = labelTotals[label]

                for f in self.features:
                    onCount = featureCounts[label][f]
                    # Laplace smoothing denom = N_y + 2k
                    probOn = (onCount + k) * 1.0 / (N_y + 2.0 * k)
                    cond[f] = probOn

                conditionalProb[label] = cond

        # temporarily set these parameters and evaluate on validation
            self.conditionalProb = conditionalProb
            guesses = self.classify(validationData)

            correct = 0
            for i in range(len(validationLabels)):
                if guesses[i] == validationLabels[i]:
                    correct += 1
            accuracy = float(correct) / len(validationLabels)

        # update best k 
            if accuracy > bestAccuracy or (accuracy == bestAccuracy and (bestK is None or k < bestK)):
                bestAccuracy = accuracy
                bestK = k
                bestConditionalProb = conditionalProb

        # best k and conditional probabilities
        self.k = bestK
        self.conditionalProb = bestConditionalProb







    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"

        for label in self.legalLabels:
        # how common is this number 
            logProb = math.log(self.prior[label])

        # add log prob for each feature
            for f in self.features:
                probOn = self.conditionalProb[label][f]
                value = datum[f]    # 0 or 1

                if value > 0:
                    p = probOn
                else:
                    p = 1.0 - probOn

                logProb += math.log(p)

            logJoint[label] = logProb

        return logJoint


    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"

        odds = util.Counter()

        eps = 1e-5  # small constant to avoid zero denom

        for f in self.features:
            p1 = self.conditionalProb[label1][f]
            p2 = self.conditionalProb[label2][f]
            odds[f] = (p1 + eps) / (p2 + eps)

        # Sort features by odds value, descending order
        sorted_features = sorted(odds.items(), key=lambda item: item[1], reverse=True)

        # Take the top 100 
        featuresOdds = [f for (f, value) in sorted_features[:100]]


        return featuresOdds
