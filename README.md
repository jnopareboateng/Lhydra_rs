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


## Synthetic Dataset Refinement

To ensure our synthetic dataset closely mimics real-world data, we conducted several analysis and refinement steps:

1. Audio Feature Acquisition
   - Extracted correct audio features from Spotify's Million Song Dataset
   - Integrated these features into our synthetic dataset

2. Genre Consistency Check
   - Identified and resolved genre inconsistencies across the dataset
   - Standardized genre labels to ensure accurate categorization

3. Date Value Imputation
   - Located entries with missing date values
   - Implemented appropriate imputation techniques to fill gaps

4. Statistical Validation
   We performed statistical tests to validate our dataset, focusing on the following hypothesis:
   
   H₀: Music preferences vary by user demographics
   H₁: Music preferences do not vary by user demographics

   (Note: Results of these tests to be added upon completion)

5. Demographic Analysis
   - Created new features based on demographic data, such as age_group
   - Analyzed relationships between individual demographic components (age, gender) and music preferences

## Recommender System Development

For our recommender system, we chose to use TensorFlow Recommenders, a library built on top of TensorFlow specifically for recommender systems.

### Model Architecture

We developed a multitask model incorporating:
- User demographics as user embeddings
- Audio information
- Other metadata (audio features)

### Model Objectives

1. Provide personalized recommendations by leveraging:
   - User demographics
   - Audio information
   - Other relevant metadata

2. Accurately predict the number of plays for each song to improve ranking

### Implementation Challenges

During the development process, we encountered several challenges:

1. Version Compatibility
   - Issue: Errors with TensorFlow v16 and higher
   - Solution: Downgraded to TensorFlow v15

2. Data Type Conversion
   - Issue: Errors in data type handling
   - Solution: Implemented preprocessing steps to convert inputs before feeding into the model

### Current Status and Next Steps

We're currently facing an issue with empty batches during training and evaluation. This is unexpected as our dataset has no missing values. We're investigating the root cause of this problem.

Next steps include:
- Resolving the empty batch issue
- Fine-tuning model performance
- Conducting comprehensive testing and validation
- Documenting final model architecture and performance metrics
