import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("data/music_data.csv")
# Assuming df is your DataFrame and 'music' is the target variable
X = df.drop('music', axis=1)
y = df['music']

# Identify categorical columns (example method, adjust based on your data)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Define a transformer that applies OneHotEncoder to the categorical columns and StandardScaler to the numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(exclude=['object', 'category']).columns),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Apply the transformations
X_transformed = preprocessor.fit_transform(X)

# Proceed with PCA on X_transformed
pca = PCA().fit(X_transformed)

# Plotting the cumulative variance explained
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid()
plt.show()
