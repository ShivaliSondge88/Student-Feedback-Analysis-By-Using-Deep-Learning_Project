# Student-Feedback-Analysis-By-Using-Deep-Learning_Project
# Student Feedback Sentiment Analysis using LSTM and Word2Vec

## Project Overview
This project focuses on building a deep learning model to analyze student feedback and classify it into positive, neutral, or negative sentiments. The process involves comprehensive text preprocessing, generating word embeddings using Word2Vec (Skip-gram), and training a Long Short-Term Memory (LSTM) neural network.

## Dataset
The dataset used for this project is `finalDataset0.2.xlsx`, extracted from `archive (30).zip`. It contains student feedback categorized across various aspects such as teaching, course content, examination, lab work, library facilities, and extracurricular activities, along with a corresponding sentiment score (-1.0 for negative, 0.0 for neutral, and 1.0 for positive).

## Project Structure & Steps

### 1. Unzip and Load Data
-   The initial step involved unzipping the provided `archive (30).zip` file to access the `finalDataset0.2.xlsx` Excel file.
-   The Excel file was then loaded into a pandas DataFrame.
-   Relevant columns containing feedback text and sentiment scores were consolidated into a single DataFrame, `consolidated_df`.
-   Numerical sentiment scores (e.g., 0.0, 1.0, -1.0) were mapped to descriptive labels ('neutral', 'positive', 'negative').

### 2. Text Preprocessing and POS Tagging
-   The 'text' column underwent thorough cleaning:
    -   Ensured all entries were string type.
    -   Removed empty or 'nan' string entries.
-   NLTK libraries were used to perform:
    -   **Tokenization**: Breaking down text into individual words.
    -   **Lowercasing**: Converting all words to lowercase.
    -   **Punctuation Removal**: Eliminating punctuation marks.
    -   **Stop Word Removal**: Filtering out common words (e.g., 'the', 'is', 'a') that do not carry significant meaning.
    -   **Part-of-Speech (POS) Tagging**: Assigning grammatical tags (e.g., noun, verb, adjective) to each word. The processed words with their POS tags were stored in the 'processed_text' column.

### 3. Word2Vec Embeddings (Skip-gram)
-   A corpus of cleaned words was extracted from the 'processed_text' column.
-   A Word2Vec model was trained using the Skip-gram architecture with the following parameters:
    -   `vector_size=100`: Each word is represented by a 100-dimensional vector.
    -   `window=5`: Maximum distance between the current and predicted word.
    -   `min_count=1`: Considers all words with a frequency of at least 1.
    -   `sg=1`: Specifies the Skip-gram training algorithm.
-   A vocabulary and word-to-index mapping were created.
-   An embedding matrix was constructed, initializing word embeddings for the LSTM layer. Words not present in the Word2Vec model were assigned random vectors.

### 4. Prepare Data for LSTM
-   The 'processed_text' entries were converted into sequences of numerical indices using the `word_to_index` mapping.
-   These sequences were padded to a uniform `max_sequence_length` (determined by the 95th percentile of sequence lengths, which was 12) using `tensorflow.keras.preprocessing.sequence.pad_sequences`.
-   The sentiment labels ('negative', 'neutral', 'positive') were one-hot encoded using `sklearn.preprocessing.LabelEncoder` and `OneHotEncoder`.
-   The data was split into training and testing sets (80% training, 20% testing).

### 5. Build and Train LSTM Model for Sentiment Analysis
-   A Sequential Keras model was built with the following layers:
    -   **Embedding Layer**: Initialized with the pre-trained Word2Vec `embedding_matrix`. This layer was set to `trainable=False` to keep the embeddings fixed.
    -   **First LSTM Layer**: 128 units with `return_sequences=True`.
    -   **First Dropout Layer**: 0.2 dropout rate to prevent overfitting.
    -   **Second LSTM Layer**: 64 units (without `return_sequences=True` as it's the last LSTM layer before the dense output).
    -   **Second Dropout Layer**: 0.2 dropout rate.
    -   **Dense Output Layer**: With `num_classes` (3) units and a 'softmax' activation for multi-class classification.
-   The model was compiled using the 'adam' optimizer and 'categorical_crossentropy' loss function, with 'accuracy' as the metric.
-   The model was trained for 20 epochs with a batch size of 32.

### 6. Evaluate Model Performance
-   The trained model was evaluated on the test set (`X_test`, `y_test`).
-   Key metrics, including **Test Loss** and **Test Accuracy**, were reported.
-   A **Classification Report** was generated to show precision, recall, and F1-score for each sentiment class ('negative', 'neutral', 'positive').
-   A **Confusion Matrix** was created and visualized as a heatmap to provide a clear understanding of the model's classification performance across different classes.

### 7. Manual Input Prediction Function
-   A Python function `predict_sentiment_from_text(text)` was created.
-   This function takes a raw text string as input, applies all the preprocessing steps (tokenization, stop word removal, POS tagging, indexing, padding), and then uses the trained LSTM model to predict the sentiment.
-   It outputs the input text, processed words, indexed words, padded sequence, prediction probabilities, and the final predicted sentiment label.

## Model Performance and Insights

**Overall Performance:** The LSTM model achieved a test accuracy of approximately 72.40%. While this seems reasonable, a deeper look into the classification report revealed significant class imbalance and biased performance.

**Class-Specific Performance:**
-   **Positive Sentiment**: The model performed exceptionally well for positive feedback, with high precision (0.76), recall (0.95), and F1-score (0.85).
-   **Neutral and Negative Sentiments**: The model struggled significantly with these classes. The precision, recall, and F1-scores for 'negative' and 'neutral' sentiments were very low (e.g., F1-scores of 0.10 for negative and 0.06 for neutral). This indicates that the model frequently misclassified these sentiments, often predicting them as 'positive'.

**Root Cause Analysis:** The primary reason for the skewed performance is likely the severe **class imbalance** in the training data, where positive samples heavily outnumbered neutral and negative ones.

## Usage

To use the trained sentiment analysis model, follow these steps:

1.  **Clone this repository** (or copy the relevant code).
2.  **Ensure all dependencies are installed** (e.g., `pandas`, `numpy`, `nltk`, `gensim`, `tensorflow`, `scikit-learn`, `seaborn`, `matplotlib`). You can typically install them using `pip`:
    ```bash
    pip install pandas numpy nltk gensim tensorflow scikit-learn seaborn matplotlib
    ```
3.  **Download NLTK data** if not already present (punkt, stopwords, averaged_perceptron_tagger, punkt_tab):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    ```
4.  **Run the entire notebook** to preprocess the data, train the Word2Vec model, build and train the LSTM model.
5.  **Use the `predict_sentiment_from_text` function** for manual sentiment prediction:

    ```python
    # Example usage of the prediction function
    new_feedback = "The teaching quality is very bad and outdated."
    predict_sentiment_from_text(new_feedback)

    another_feedback = "The lab facilities are adequate."
    predict_sentiment_from_text(another_feedback)
    ```

## Next Steps and Improvements

-   **Address Class Imbalance**: Implement techniques like oversampling (e.g., SMOTE), undersampling, or using class weights in the loss function to improve the model's ability to recognize minority classes.
-   **Hyperparameter Tuning**: Experiment with different LSTM units, dropout rates, batch sizes, and epochs to optimize model performance.
-   **Explore Advanced Architectures**: Consider using Bidirectional LSTMs (Bi-LSTM) or Transformer-based models for potentially better contextual understanding.
-   **Augment Data**: Generate synthetic text data for underrepresented classes to balance the dataset.
-   **Aspect-Based Sentiment Analysis**: For a more granular analysis, explicitly label aspects within the feedback (e.g., 'teaching: positive', 'lab work: negative') and train an aspect-based sentiment model.
