# Optimizing an ML Pipeline in Azure

## Table of Contents

- [Optimizing an ML Pipeline in Azure](#optimizing-an-ml-pipeline-in-azure)
  * [Overview](#overview)
  * [Summary](#summary)
  * [Scikit-learn Pipeline](#scikit-learn-pipeline)
  * [AutoML](#automl)
  * [Pipeline comparison](#pipeline-comparison)
  * [Future work](#future-work)
  * [Proof of cluster clean up](#proof-of-cluster-clean-up)



## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, I build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

This is a subset of a standard machine learning dataset with data about the direct marketing campaigns of a Portuguese banking institution. It provides 20 independent variables, both numerical and categorical, about bank clients, and a categorical dependant variables: has the client subscribed a term deposit? (binary: 'yes','no'). This last one variable is what I try to predict in this classifcation excersise.

The best performing model I found was a VotingEnsemble, generated in the AutoML part of the AzureML experiment, that got a 0.9161 of accuracy. It must be said that best models were all in the range between 0.911 and the above mentioned 0.916.


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

I began with a stand-alone Python script, implementing a Scikit_learn model of logistic regression, a classical classification algorithm. 

Data were loaded from an internet adress (url), cleaned, transformed and divides in training and test sets.

In order to run the algortihm, two parameters were defined:

- C: a positive float number, which is the Inverse of regularization strength, and was set to a default of 1 (the greater the vaue, the lesser the regularization). 

-max_iter: a positive integer, meaning the number of iterations the algortihm is allowed to run before convergence (default set to 100).

Aferwards, this same script was uploaded to an AzureML notebook space, and a notebook was used to run an HyperDrive experiment, trying to get, automatically, the best model parameters for the original script.

**What are the benefits of the parameter sampler you chose?**

Since previous trials with the original script had shown me that neither trying several values of C nor of max_iter got much effect on accuracy, I discarded a Grid Parameter Sampling, and opted for a Random Parameter Sampler, so HyperDrive could choose, randomly, different values of C, and combine them whit a certain numer of values of max_iter, rangng from 100 (default) and 1000 (since no greater values was able to offer better results with the original script).

I was shy to use the Bayesian Parameter Sampling, since I quite don't understand yet how it works. In retrospective, it's something I should have given a try. And I'll do.

**What are the benefits of the early stopping policy you chose?**

I choose a BanditPolicy, since the trials with the original script in a Python envoronment had shown me that there were little gains in accuracy from run to run, even with wid changing parameter values. Thus, I defined a slack factor of 0.1 every 2 runs, something I look now as a to tight a criterium.


## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

In the given time, the AutoML experiment run 37 diferent models, the best being a Voting Ensemble model, with an accuracy of 0.91611.

Votig Ensemble is a class that can't be invoqued directly, but can be specified in the AutoMLConfig instance.

In this case, it's a default AutoML uses once it has run several models. If I understood this right, this implementation uses a soft voting mechanism, summing up the predicted probabilities for class labels in the previous tested models, and predicting the class label with the largest sum probability.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

As said, the AutoML Votign Ensemble model got an accuray of 0.91611, against an accuracy of 0.91163 of the HyperDrive model, and 0.91102 of the original script. So, the gain was a meager one, in the order of the thousandths.

AutoML is a better architecture than that provided by HyperDrive, since whereas this last one gets the original model for a given and only tries to optimize its parametes, AutoML takes hold of a comkbination of scalers (MaxAbsScaler, SparseNormalizer and StandardScalarWraper)
and models (ExtremeRandomTrees, LightGBM, RandomForest, SGD, XGBoostClassifier, as weel as Voting and Stack Ensembles) to look for a better model.


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

Since this is an unbalanced data set, perhaps accuracy is not the best criterium to optimize, and AUC or weighted AUC woukd be better performance metrics.

Are there other classification algorithms that would merit a try?

Could different parameter sampling methods and policies bring a better result?

And last, but ot least: could a greater amount of minutes for timeout get a better result?


## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

Uh, uh. Sorry. I'll get that fixed.

