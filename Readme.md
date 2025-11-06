Here's the updated README with your requested additions:[1][2][9][11]

***

# Twitter Sentiment Analysis

A comprehensive sentiment analysis system that classifies tweets into four sentiment categories (Positive, Negative, Neutral, Irrelevant) using deep learning and traditional machine learning approaches with DVC for data and model versioning.

## Overview

This project performs sentiment analysis on Twitter data using both traditional ML algorithms and deep learning models (LSTM and GRU). The system processes ~75,000 tweets and achieves over 80% accuracy using ensemble methods. Interestingly, traditional models outperform deep learning approaches on this dataset, demonstrating that simpler models can be more effective for moderately-sized, less complex datasets.

## Dataset

- **Source Files**: `twitter_training.csv` and `twitter_validation.csv`
- **Total Tweets**: 74,994 (after preprocessing)
- **Sentiment Classes**: Positive, Negative, Neutral, Irrelevant
- **Topics Covered**: Gaming (Borderlands, CS-GO, GTA), Tech Companies (Microsoft, Nvidia), and more
- **Train-Test Split**: 80-20 stratified split
- **Version Control**: Tracked with DVC (`data.dvc`)

## Features

### Text Preprocessing Pipeline
- Tokenization using NLTK TweetTokenizer
- Lowercasing and stopword removal
- Lemmatization with WordNet
- Punctuation and special character removal
- TF-IDF vectorization for traditional ML models

### Models Implemented

#### Deep Learning Models
- **LSTM Networks**: Multiple variants with different architectures
  - Best validation accuracy: ~81%
  - Architecture: 2-layer LSTM with dropout and recurrent dropout
- **GRU Networks**: Multiple variants with regularization
  - Best validation accuracy: ~81%
  - Architecture: 3-layer GRU with dropout regularization
- **Text Vectorization**: Custom vocabulary (10,000 tokens, max length 42)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Tracked Models**: Two best LSTM models versioned with DVC

#### Traditional ML Models
- **Random Forest**: Best performer (~88% test accuracy)
- **Decision Tree**: ~88% test accuracy with max_depth=85
- **Logistic Regression**: Baseline model (~74% test accuracy)
- **Tracked Models**: Logistic Regression and Random Forest versioned with DVC (`models.dvc`)

## Technologies Used

- **Python 3.11**
- **Deep Learning**: TensorFlow/Keras
- **ML Libraries**: scikit-learn
- **NLP**: NLTK
- **Version Control**: DVC (Data Version Control)
- **Experiment Tracking**: MLflow
- **Data Handling**: pandas, numpy
- **Model Persistence**: joblib

## Installation

```bash
pip install tensorflow scikit-learn nltk pandas numpy mlflow dvc
```

Download NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

Initialize DVC:
```bash
dvc init
dvc remote add -d storage <your-remote-storage>
```

## Usage

1. **Pull Data and Models from DVC**:
```bash
dvc pull data.dvc
dvc pull models.dvc
```

2. **Data Preprocessing**:
```python
# Load and preprocess data
combined_df = pd.concat([train_df, test_df])
combined_df['processed_text'] = preprocess_pipeline(combined_df['text'])
```

3. **Train Deep Learning Model**:
```python
# LSTM/GRU model training with MLflow tracking
model.fit(train_inputs, train_target, 
          epochs=20, 
          validation_data=(val_inputs, val_target),
          callbacks=[early_stopping, reduce_lr])
```

4. **Train Traditional ML Model**:
```python
# Random Forest training
model_forest = RandomForestClassifier(n_estimators=120, max_depth=250)
model_forest.fit(train_tfidf, train_label)
```

5. **Version Control with DVC**:
```bash
# Track new data or models
dvc add Data/twitter_training.csv
dvc add models/rf_model.pkl
git add data.dvc models.dvc
git commit -m "Update models and data"
dvc push
```

## Model Performance

| Model | Train Accuracy | Test Accuracy | Test F1-Score |
|-------|----------------|---------------|---------------|
| Random Forest v5 | 94.78% | 88.25% | 88.41% |
| Decision Tree v3 | 94.78% | 88.25% | 88.41% |
| LSTM v2 (64-64) | 90.43% | 81.05% | 80.99% |
| GRU v2 (32-32-32) | 88.83% | 81.49% | 81.00% |
| Logistic Regression | 81.40% | 74.44% | 74.53% |

## Why Traditional Models Outperform Deep Learning

Traditional machine learning models (Random Forest, Decision Tree) significantly outperform deep learning approaches (LSTM, GRU) in this project due to several key factors:

1. **Dataset Size**: With ~75,000 tweets, the dataset is not large enough for deep learning models to leverage their full potential. Deep learning models typically require hundreds of thousands to millions of samples to learn complex patterns effectively.

2. **Limited Complexity**: The sentiment patterns in tweets are relatively straightforward and can be captured well by traditional feature engineering (TF-IDF) combined with ensemble methods. Deep learning excels at capturing intricate sequential dependencies and hierarchical representations, which aren't necessary for this task.

3. **Feature Representation**: TF-IDF vectorization combined with Random Forest's ability to handle high-dimensional sparse features works exceptionally well for text classification on moderate-sized datasets.

4. **Computational Efficiency**: Traditional models train faster and require less computational resources while achieving superior results, making them more practical for this specific use case.

5. **No Sequential Dependencies**: While tweets contain text, the sentiment classification task doesn't heavily rely on long-range sequential dependencies that LSTMs are designed to capture. The "bag-of-words" approach with TF-IDF is sufficient.

This demonstrates an important principle in ML: **more complex models aren't always better**. Model selection should be based on dataset characteristics, complexity of the task, and available computational resources.

## Project Structure

```
├── Data/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── Models/
│   ├── lstm.h5            # LSTM models (DVC tracked)
│   ├── rf_model.pkl       # Random Forest (DVC tracked)
│   └── logreg_model.pkl   # Logistic Regression (DVC tracked)
├── Twitter-Sentimental.ipynb
├── data.dvc               # DVC pointer for dataset
├── models.dvc             # DVC pointer for models
├── .dvc/                  # DVC configuration
└── mlruns/                # MLflow experiment tracking
```

## Version Control Strategy

### DVC Tracked Assets
- **data.dvc**: Points to `twitter_training.csv` and `twitter_validation.csv`
- **models.dvc**: Points to versioned models:
  - Logistic Regression baseline
  - Random Forest (best performer)
  - Two best LSTM variants

### Benefits of DVC Integration
- Reproducible experiments across team members
- Version control for large data files and models
- Easy rollback to previous model versions
- Efficient storage with remote backends
- Seamless collaboration without storing large binaries in Git

## Key Insights

- **Traditional models outperform deep learning** on this moderately-sized dataset due to limited complexity and insufficient data for deep models to capture intricate sequential patterns
- Text preprocessing (lemmatization, stopword removal) significantly improves performance
- Deep learning models show signs of overfitting despite regularization techniques
- Class balance is relatively good, reducing need for resampling techniques
- TF-IDF + Random Forest provides an excellent baseline that's hard to beat for this task

## Experiment Tracking

All experiments are tracked using MLflow, including:
- Model hyperparameters
- Training/validation metrics
- Model artifacts
- Tags for easy filtering
- Comparison across traditional and deep learning approaches

## Future Improvements

- Implement transfer learning with pre-trained embeddings (Word2Vec, GloVe)
- Experiment with transformer-based models (BERT, RoBERTa) for larger datasets
- Add cross-validation for more robust evaluation
- Deploy model as REST API using DVC pipeline
- Implement real-time tweet classification
- Expand dataset size to better leverage deep learning capabilities
- Create DVC pipelines for automated retraining

## Requirements

```txt
tensorflow>=2.13.0
scikit-learn>=1.3.0
nltk>=3.8.1
pandas>=2.0.0
numpy>=1.24.0
mlflow>=2.5.0
dvc>=3.0.0
joblib>=1.3.0
```

## License

This project is open-source and available for educational purposes.

---