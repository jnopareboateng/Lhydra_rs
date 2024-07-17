# LHydra Hybrid Recommender System

## Introduction
In this project, we introduce LHydra, a demographic-aware hybrid recommender system that leverages collaborative filtering, content-based filtering, and reinforcement learning to provide personalized recommendations.

## Dataset
We have generated a synthetic dataset comprising 350K unique items to ensure high data quality and the ability to simulate real-world experiences. During the quality assurance process, we identified a mismatch in name and gender pairs.

## Model Development
To address this, we developed a predictive model for name-to-gender classification using the following algorithms:
- Multinomial Naive Bayes 
- Random Forests Classifier (RFC)
- Support Vector Classifier (SVC)
- Gradient Boosting Classifier

The choice of algorithms was based on their proven track record in classification tasks.

## Results
The initial training times for the models on base parameters were approximately 119 minutes. After hyperparameter tuning, the process extended to over 32 hours.

The accuracy scores for the models with base parameters are as follows:

| Algorithm          | Accuracy Score |
|--------------------|----------------|
| Naive Bayes        | 0.7591         |
| Random Forest      | 0.8275         |
| SVC                | 0.8635         |
| Gradient Boosting  | 0.7676         |

## Decision
Despite SVC having the highest accuracy, its training time was significantly longer than the others. We chose the Random Forest Classifier for its balance between accuracy (82.7%) and efficiency. The 4% difference from SVC is within the acceptable margin of error (5%), making it a suitable choice for our gender prediction tasks.
