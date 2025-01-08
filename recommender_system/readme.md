# Music Recommendation System Documentation

## 1. Project Overview

**Project Name:** Hybrid Deep Learning Music Recommendation System

**Project Lead:** [Insert Name and Contact Information]

**Team Members:** [List Team Members and Roles]

**Start Date:** [Insert Start Date]

**End Date:** [Insert End Date]

**Objective:** To develop a personalized music recommendation system that leverages both user interaction data and music content features, using a hybrid deep learning architecture to enhance recommendation accuracy and diversity.

## 2. Problem Statement

### 2.1 Background

The proliferation of digital music platforms has led to an overwhelming amount of music content available to users. Traditional recommendation methods often struggle to capture the nuanced preferences of individual users and the diverse characteristics of music. This project addresses the need for a more sophisticated recommendation system that can effectively personalize music suggestions.

### 2.2 Specific Problem

The project aims to solve the problem of generating accurate and diverse music recommendations by effectively modeling user preferences and music characteristics via demographic factors. This involves addressing challenges such as:

*   **Cold Start Problem:**  Effectively recommending music to new users or for newly released songs with limited interaction data.
*   **Data Sparsity:** User-item interaction matrices are often sparse, making it difficult to identify meaningful patterns.
*   **Serendipity and Diversity:**  Balancing the need to recommend relevant music with the desire to introduce users to new artists and genres they might not have discovered otherwise.

### 2.3 Goals

The primary goals of this project are:

*   Develop a hybrid recommendation system that combines collaborative filtering and content-based approaches.
*   Utilize deep learning techniques to model complex user-item interactions and music features.
*   Implement a robust data preprocessing pipeline to handle diverse data types and ensure data quality.
*   Evaluate the system's performance using appropriate ranking and diversity metrics.
*   Create a scalable and efficient system capable of handling large datasets and real-time recommendations.

## 3. Data Collection and Preprocessing

### 3.1 Data Sources

The project utilizes a combination of data sources:

*   **User Interaction Data:**  This includes user listening history, play counts, and user demographics (age, gender).
*   **Music Metadata:**  Information about music tracks, including artist names, genres, release dates, and track titles.
*   **Audio Features:**  Acoustic properties of music tracks extracted from the Spotify API, such as acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, tempo, and time signature.

### 3.2 Data Description

The raw dataset comprises various data types:

*   **User Data:**
    *   `user_id` (int): Unique identifier for users.
    *   `age` (int): User age.
    *   `gender` (str): User gender (M/F).
*   **Music Data:**
    *   `music` (str):  Track name.
    *   `artist_name` (str): Primary artist.
    *   `featured_artists` (str):  Collaborating artists.
    *   `genre` (str): Music genre.
    *   `plays` (int): Number of plays.
    *   `duration` (float): Track duration in minutes.
    *   `release_date` (str): Track release date.
*   **Audio Features:**
    *   `acousticness` (float):  A measure of whether the track is acoustic.
    *   `danceability` (float): Describes how suitable a track is for dancing.
    *   `energy` (float):  A measure of intensity and activity.
    *   `instrumentalness` (float): Predicts whether a track contains no vocals.
    *   `liveness` (float):  Presence of an audience in the recording.
    *   `loudness` (float): The overall loudness of a track in decibels (dB).
    *   `speechiness` (float):  Presence of spoken words in a track.
    *   `valence` (float):  Musical positiveness.
    *   `tempo` (float):  Track tempo in beats per minute (BPM).
    *   `time_signature` (int): Estimated time signature.
    *   `key` (int): The key the track is in.
    *   `mode` (int): Indicates the modality (major or minor) of a track.
    *   `explicit` (int): Indicates whether the track contains explicit lyrics.

### 3.3 Preprocessing Steps

Data preprocessing is handled by the `MusicPreprocessor` class and includes the following steps:

1. **Initial Data Cleaning:**
    *   Basic data validation and type checking.
    *   Handling of missing values (imputation or removal based on feature).
    *   Outlier treatment using RobustScaler or IQR-based capping as specified.
    *   Logging of all data transformation steps for reproducibility.
2. **Feature Engineering:**
    *   **Temporal Features:**
        *   `release_year`:  Extracted from `release_date`.
        *   `music_age`: Calculated as  `current_year - release_year`.
        *   `age_group`:  Categorization of users into age groups (e.g., Gen Z, Gen X).
    *   **Interaction Features:**
        *   `energy_loudness`:  Product of `energy` and `loudness`.
        *   `dance_valence`:  Product of `danceability` and `valence`.
        *   `plays_log`: Logarithmic transformation of `plays` (`np.log1p(df_processed["plays"])`) to handle skewed distribution.
    *   **Time Signature Handling:** Filtering time signatures to include only expected values (3 and 4).
3. **Data Transformation:**
    *   **Numerical Features:**
        *   Robust Scaling using `sklearn.preprocessing.RobustScaler` to handle outliers.
        *   Log transformation for skewed features: `plays`, `duration`, `tempo`.
    *   **Categorical Features:**
        *   Label Encoding for `gender` using `sklearn.preprocessing.LabelEncoder`.
        *   TF-IDF Vectorization for text features: `artist_name`, `genre`, `featured_artists` using `sklearn.feature_extraction.text.TfidfVectorizer`.
    *   **Audio Features:** Standard scaling or robust scaling depending on the outlier handling strategy.

        ```python
        NUMERICAL_COLS = [
            "age", "duration", "acousticness", "key", "mode",
            "speechiness", "instrumentalness", "liveness",
            "tempo", "time_signature", "explicit",
            "music_age", "energy_loudness", "dance_valence"
        ]
        ```
4. **Data Splitting:**  Stratified splitting of the dataset into training, validation, and test sets.

#### 3.3.1 MusicPreprocessor Class

The `MusicPreprocessor` class encapsulates the logic for data cleaning, outlier handling, feature scaling, and feature engineering.

**Class Definition:**

```python
class MusicPreprocessor:
    def __init__(self, handle_outliers="robust"):
        """
        Initialize the preprocessor with specified outlier handling strategy.

        Parameters:
            handle_outliers (str): Strategy for handling outliers
                - 'robust': Uses RobustScaler from sklearn
                - 'cap': Uses IQR-based capping
                - 'none': Leaves outliers unchanged
        """
        # ... (Initialization of scalers, encoders, etc.)
```

**Key Methods:**

*   `fit_transform(self, df)`: Fits the preprocessor to the input DataFrame and transforms it.
*   `transform(self, df)`: Transforms new data using the fitted preprocessor parameters.
*   `_cap_outliers(self, series)`:  Caps outliers in a series using the IQR method.
*   `_handle_time_signature_outliers(self, df)`:  Filters time signatures to include only expected values (3 and 4).
*   `load_preprocessors(self, preprocessor_path)`: Loads pre-fitted preprocessors from a specified path.
*   `save_preprocessors(self, preprocessor_path)`: Saves fitted preprocessors to a specified path.

**Visualization Support:**

The `MusicPreprocessor` class also includes visualization capabilities to analyze data distributions before and after transformations, such as:

*   Play count distributions (original vs. log-transformed).

*   Energy feature distributions (original vs. transformed).
*   Time signature distributions (original vs. processed).

#### 3.3.2 Data Quality Metrics and Validation

*   **Numeric Range Validation:** Ensure numerical features fall within expected ranges.
*   **Outlier Detection:**  Identify and handle outliers using chosen methods.
*   **Missing Value Tracking:** Monitor and address missing values appropriately.
*   **Data Type Consistency:** Enforce consistent data types for each feature.
*   **Feature Statistics:** Calculate descriptive statistics (mean, median, standard deviation, etc.) to understand feature distributions.
*   **Correlation Analysis:** Analyze feature correlations to identify potential redundancies or interactions.

## 4. Model Selection and Training

### ### 4.1 Model Selection

A hybrid recommendation system is chosen, combining collaborative filtering and content-based approaches within a deep learning framework. This architecture is selected for its ability to:

*   **Capture Complex Relationships:** Deep learning models can learn non-linear relationships between user preferences and item characteristics.
*   **Leverage Diverse Data:** The hybrid approach effectively integrates user interaction data, music metadata, and audio features.
*   **Address Cold Start Problem:** Content-based features help provide recommendations for new users or items with limited interaction history.
*   **Personalize Recommendations:** The model learns personalized embeddings for users and items, capturing individual tastes and preferences.

### 4.2 HybridRecommender Neural Network

The core of the recommendation system is the `HybridRecommender` neural network, which consists of the following components:

1. **Embedding Layers:**
    *   `self.user_embedding = nn.Embedding(num_users, embedding_dim)`:  Maps user IDs to dense embedding vectors.
    *   `self.gender_embedding = nn.Embedding(num_genders, embedding_dim)`:  Maps gender to dense embedding vectors.
    *   `self.release_year_embedding = nn.Embedding(num_release_years, embedding_dim)`:  Maps release year to dense embedding vectors.

2. **Feature Transformations:**
    *   `self.music_fc = nn.Sequential(...)`:  Transforms music features using a fully connected network with batch normalization and ReLU activation. This processes one-hot encoded music items.

        ```python
        self.music_fc = nn.Sequential(
            nn.Linear(num_music_items, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        ```

3. **Deep Neural Network:**
    *   Multiple residual blocks with skip connections to facilitate learning of complex patterns and prevent vanishing gradients. Each block consists of:
        *   Linear Layer
        *   Batch Normalization
        *   ReLU Activation
        *   Dropout Layer for regularization.
    *   The output of each residual block is added to its input (skip connection).
    *   The outputs of embedding layers and feature transformation layers are concatenated and fed into the deep neural network.
    *   Final output layer with a single neuron for predicting the relevance score of a music item for a user.

### 4.3 Loss Function

#### 4.3.1 EnhancedListNetLoss

A custom loss function, `EnhancedListNetLoss`, is implemented to optimize the model. This loss function combines:

1. **ListNet Ranking Loss:** Encourages the model to learn the correct ranking of items by minimizing the Kullback-Leibler divergence between predicted and true probability distributions over item rankings.

    ```python
    P_y_pred = F.softmax(y_pred, dim=0)
    P_y_true = F.softmax(y_true, dim=0)
    listnet_loss = -torch.sum(P_y_true * torch.log(P_y_pred + 1e-10))
    ```

2. **Intra-List Similarity (ILS) Regularization:** Promotes diversity in the recommended list by penalizing recommendations that are too similar to each other. The similarity can be based on genre, artist, or other music features.

    ```python
    similarity_matrix = self.compute_similarity_matrix(combined_features)
    ils_penalty = self.compute_ils_penalty(similarity_matrix, y_pred)
    ```

3. **Temperature Scaling:**  A temperature parameter is used to control the sharpness of the probability distributions in both the ListNet loss and ILS regularization.

**Loss Function Implementation:**

```python
def forward(self, y_pred, y_true, genre_features, artist_features, music_features):
    # ListNet loss
    P_y_pred = F.softmax(y_pred, dim=0)
    P_y_true = F.softmax(y_true, dim=0)
    listnet_loss = -torch.sum(P_y_true * torch.log(P_y_pred + 1e-10))
    
    # Combine features for similarity computation
    combined_features = torch.cat([genre_features, artist_features, music_features], dim=1)
    
    # ILS penalty
    similarity_matrix = self.compute_similarity_matrix(combined_features)
    ils_penalty = self.compute_ils_penalty(similarity_matrix, y_pred)
    
    return listnet_loss + self.ils_weight * ils_penalty
```

### 4.4 Training Process

The model is trained using the following configuration:

```python
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 64
NUM_EPOCHS = 20
EMBEDDING_DIM = 32
LEARNING_RATE = 0.01
```

**Training Loop Features:**

1. **Optimizer:**  Adam optimizer is used for updating the model weights.
2. **Early Stopping:**  Training is stopped if the validation loss does not improve for a specified number of epochs (patience) to prevent overfitting.
3. **Learning Rate Scheduling:**  The learning rate is dynamically adjusted during training (e.g., using a ReduceLROnPlateau scheduler) to improve convergence.
4. **Gradient Checking:**  Gradient clipping is applied to prevent exploding gradients.
5. **Model Checkpointing:**  The best performing model (based on validation loss) is saved during training.
6. **Progress Tracking:**  A progress bar (using `tqdm`) is displayed during training to monitor progress.
7. **Comprehensive Logging:**  Training and validation metrics, as well as other relevant information, are logged during training for monitoring and debugging.

**Training Procedure:**

1. **Data Loading:**  Data is loaded in batches using PyTorch `DataLoader` objects.
2. **Forward Pass:**  The model computes predicted relevance scores for each user-item pair in a batch.
3. **Loss Calculation:**  The `EnhancedListNetLoss` is calculated based on predicted scores and true labels, as well as genre, artist, and music feature embeddings for ILS regularization.
4. **Backward Pass:**  Gradients are computed using backpropagation.
5. **Optimization Step:**  Model weights are updated using the optimizer and calculated gradients.
6. **Evaluation:**  The model is evaluated on the validation set after each epoch to monitor performance and detect overfitting.
7. **Checkpointing and Early Stopping:**  The model is saved if it achieves the best validation loss so far, and training is stopped early if the validation loss does not improve for a specified number of epochs.

### 4.5 Libraries and Tools

The following libraries and tools are used in this project:

*   **Python:** The primary programming language.
*   **PyTorch:**  Deep learning framework.
*   **Pandas:**  Data manipulation and analysis.
*   **NumPy:**  Numerical computing.
*   **Scikit-learn:**  Machine learning tools for preprocessing, feature engineering, and evaluation.
*   **Tqdm:**  Library for displaying progress bars.
*   **Matplotlib and Seaborn:** Data visualization libraries.

## 5. Evaluation Metrics and Results

### 5.1 Evaluation Metrics

The performance of the recommendation system is evaluated using a combination of ranking and diversity metrics.

#### 5.1.1 Ranking Metrics

These metrics assess the accuracy of the model's recommendations by evaluating the relevance and ranking of items:

*   **Normalized Discounted Cumulative Gain (NDCG@k):**  Measures the relevance of the recommended items, considering their position in the ranked list. Higher NDCG indicates better ranking quality.
*   **Precision@k:**  The proportion of relevant items among the top-k recommendations.
*   **Recall@k:**  The proportion of relevant items that are included in the top-k recommendations.
*   **F1-Score@k:**  The harmonic mean of precision and recall, providing a balance between the two.

#### 5.1.2 Diversity Metrics

These metrics evaluate the diversity of the recommended lists, aiming to avoid recommending items that are too similar:

*   **Intra-list Similarity (ILS):** Measures the average similarity between items in a recommended list. Lower ILS indicates higher diversity. Similarity can be computed based on genre, artist, or other features.
*   **Genre Diversity:**  The number of unique genres represented in the top-k recommendations.
*   **Artist Diversity:**  The number of unique artists represented in the top-k recommendations.

### 5.2 Results

The results of the evaluation metrics are presented for the test dataset. Performance is measured for different values of `k` (e.g., k=5, 10, 20) to assess the system's ability to provide relevant recommendations at various list sizes.

**(Placeholder for actual results - insert your results here)**

| Metric           | K=10 |
| ---------------- | ---- |
| NDCG             | 0.82 |
| Precision        | 0.55 |
| Recall           | 0.45 |
| F1-Score         | 0.50 |
| ILS (Genre)      | 0.25 |
| ILS (Artist)     | 0.30 |
| Genre Diversity  | 4.5  |
| Artist Diversity | 5.0  |

**Example Visualization:**

*   **Precision-Recall Curve:** A graph showing the trade-off between precision and recall for different recommendation thresholds.
*   **NDCG vs. k:** A plot showing how NDCG changes with the number of recommended items (k).
*   **Diversity vs. k:** Plots showing how genre and artist diversity change with the number of recommended items (k).
*   **Histograms of Recommendation Scores:** Visualizing the distribution of predicted relevance scores for relevant and irrelevant items.

### 5.3 Discussion

The results demonstrate the effectiveness of the hybrid deep learning approach for music recommendation.

*   **(Ranking Performance)**: The NDCG, Precision, Recall, and F1-scores indicate that the model is able to provide relevant recommendations to users. The performance metrics at different values of `k` provide insights into the system's ability to generate accurate recommendations at various list sizes.
*   **(Diversity)**: The ILS scores and diversity metrics show that the system can generate diverse recommendations, balancing relevance with the introduction of new artists and genres. The ILS regularization in the loss function and the careful feature engineering of diversity related attributes played a crucial role in achieving these results.
*   **(Challenges)**: Challenges encountered during the project included addressing cold start issues, handling imbalanced datasets, and fine-tuning hyperparameters. Cold start was mitigated by the incorporation of content-based features and the use of diverse demographic data. Data imbalance was addressed through stratified sampling during data splitting and potential weighting in the loss function. Hyperparameter tuning was done extensively with the help of the validation dataset and techniques like grid search.
*   **(Ablation Studies)**: [Optional] Include ablation studies where you evaluate the contribution of individual components of the model (e.g., ILS regularization, specific features) to the overall performance.
*   **(Comparison with Baseline Models)**: [Optional] If applicable, compare the performance of the hybrid model with simpler baseline models, such as a purely collaborative filtering or content-based approach, to demonstrate the benefits of the hybrid architecture.

## 6. Conclusion and Future Work

### 6.1 Conclusion

This project successfully developed a hybrid deep learning music recommendation system that effectively combines collaborative filtering and content-based approaches. The system leverages user interaction data, music metadata, and audio features to generate personalized and diverse music recommendations. The use of a custom loss function with ILS regularization enhances the diversity of the recommendations, while the deep learning architecture captures complex relationships between users and items. The evaluation results demonstrate the system's ability to provide accurate and diverse recommendations, outperforming baseline models or achieving acceptable metrics.

The robust data preprocessing pipeline, including outlier handling and feature engineering, ensures data quality and contributes to the overall performance of the system. The model training process incorporates best practices such as early stopping, learning rate scheduling, and gradient checking, resulting in a stable and well-trained model.

### 6.2 Future Work

Several avenues exist for future research and improvements:

*   **Explore Advanced Deep Learning Architectures:** Investigate more sophisticated deep learning models, such as transformers or graph neural networks, to further improve recommendation accuracy and capture complex relationships in the data.
*   **Incorporate Contextual Information:** Integrate contextual information, such as time of day, location, or user mood, to provide more personalized and context-aware recommendations.
*   **Enhance Cold Start Handling:** Develop more advanced strategies for addressing the cold start problem, such as meta-learning or few-shot learning techniques.
*   **Dynamic User Embeddings:** Adapt user embeddings over time to reflect evolving music preferences.
*   **Interactive Recommendation Interface:** Develop a user-friendly interface that allows users to provide feedback on recommendations and refine their preferences interactively.
*   **Real-time Recommendations:** Implement a scalable system capable of providing real-time recommendations based on user interactions and newly released music.
*   **Experiment with Additional Audio Features:** Utilize more advanced audio analysis techniques to extract additional features that capture aspects of music such as rhythm, harmony, and timbre.
*   **Personalized Diversity:**  Tailor the level of diversity in recommendations to individual user preferences, allowing users to control the trade-off between relevance and novelty.
*   **Deployment and A/B Testing:**  Deploy the recommendation system in a real-world setting and conduct A/B testing to evaluate its performance and user satisfaction in a live environment.

## 7. References

*   List all relevant papers, articles, websites, and other resources used in the project.
*   Include citations for any code or algorithms adapted from external sources.

**Example References:**

*   Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
*   Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009, June). BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence (pp. 452-461). AUAI Press.
*   He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).
*   Spotify API Documentation: [Insert Link to Spotify API Documentation]
*   Relevant research papers on hybrid recommendation systems and diversity in recommendations.

This detailed documentation provides a comprehensive overview of the music recommendation system project, covering its objectives, methodologies, implementation details, results, and future directions. It is intended to facilitate understanding, reproducibility, and further development of the system.


