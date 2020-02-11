---
title: Home
---
# Deep Learning with TensorFlow

## Introduction to Machine Learning

Machine learning is a computer science research area that deals with methods to identify and implement systems and algorithms by which a computer can learn, based on examples given in the input.

The challenge of machine learning is to allow a computer to learn how to automatically recognize complex patterns and make decisions that are as smart as possible. The entire learning process requires a dataset as follows:

* **Training set**: this is the knowledge base used to train the machine learning algorithm. During this phase, the parameters of the machine learning model can be tuned according to the performance obtained.

* **Testing data**: this is used only for evaluating the performance of the model on unseen data.

Learning theory uses mathematical tools that are derived from **probability theory** and **information theory**. This allows you to assess the optimality of some methods over others.

There are basically three learning paradigms that will be briefly discusses:

* [Supervised learning](#supervised_learning)
* [Unsupervised learning](#unsupervised_learning)
* [Learning with reinforcement](#learning_with_reinforcement)

## <a name="supervised_learning"></a>Supervised learning

Supervised learning is an automatic learning task. It is based on a number of preclassified examples, in which the category of each of the inputs used as examples, should belong. In this case, the crucial issue is the problem of generalization. After the analysis of a sample of examples, the system should produce a model that should work well for all possible inputs.

The set consists of labeled data, that is, objects and their associated classes. This set of labeled examples, therefore, constitutes the training set.

Most of the supervised learning algorithms share one characteristic: the training is performed by the minimization of a particular loss or cost function, representing the output error provided by the system with respect to the desired possible output, because the training set provides us with what must be the desired output.

The system then changes its internal editable parameters, the weights, to minimize this error function. The goodness of the model is evaluated, providing a second set of labeled examples (the test set), evaluating the percentage of correctly classified examples and the percentage of misclassified examples.

The supervised learning context includes the classifiers, but also the learning of functions that predict numeric values. This task is the **regression**. In a regression problem, the training set is a pair formed by an object and the associated numeric value. There are several supervised learning algorithms that have been developed for classification and regression. These can be grouped into the formula used to represent the classifier or the learning predictor, among all, decision trees, decision rules, neural networks and Bayesian networks.




