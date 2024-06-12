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


## Exploratory Data Analysis (EDA)

### Data Preprocessing
Before diving into the EDA, we integrated the corrected gender predictions into the dataset, replacing the initial values. This ensures that our analysis is based on the most accurate data available.

### Feature Engineering
We performed a log transformation on the `plays` column to normalize the distribution and reduce the impact of outliers. Subsequently, we applied a MinMax scaler within the range of 1 to 5 to create a new `ratings` column. This column is pivotal for the reinforcement learning component of our system, as it provides a standardized measure of song popularity.

### Visualization and Statistical Analysis
Our approach to EDA encompasses a variety of techniques, including:
- **Descriptive Statistics**: To summarize the central tendency, dispersion, and shape of the dataset's distribution.
- **Data Visualization**: Employing plots such as histograms, box plots, and scatter plots to visually inspect the data and uncover patterns.
- **Correlation Analysis**: To identify and quantify the relationships between variables.

These methods are instrumental in gaining insights into the data's characteristics and guiding subsequent modeling decisions.

### EDA Notebook
The complete EDA process, along with the stunning visualizations and statistical analyses, is documented in the Jupyter notebook titled `eda.ipynb`, located in the root directory of the codebase. We encourage you to review this notebook and share your feedback and suggestions to further refine our analysis.

