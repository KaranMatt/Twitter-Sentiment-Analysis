
# Twitter Sentiment Analysis

A comprehensive sentiment analysis system that classifies tweets into four sentiment categories (Positive, Negative, Neutral, Irrelevant) using deep learning and traditional machine learning approaches, served via a FastAPI REST API, with DVC for data and model versioning.

## Overview

This project performs sentiment analysis on Twitter data using both traditional ML algorithms and deep learning models (LSTM and GRU). The system processes ~75,000 tweets and achieves over 80% accuracy using ensemble methods. Interestingly, traditional models outperform deep learning approaches on this dataset, demonstrating that simpler models can be more effective for moderately-sized, less complex datasets.

## Repository

[KaranMatt/Twitter-Sentiment-Analysis](https://github.com/KaranMatt/Twitter-Sentiment-Analysis)

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
- **Vocabulary Persistence**: The fitted `TextVectorization` vocabulary is extracted via `get_vocabulary()` and saved to `Models/TextVectorVocab.pkl` using joblib. At API startup, the vocab is reloaded and injected back into the vectorization layer via `set_vocabulary()`, ensuring consistent tokenization without needing to re-adapt the layer on raw data.
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Tracked Models**: Two best LSTM models versioned with DVC

#### Traditional ML Models
- **Random Forest**: Best performer (~88% test accuracy), saved as `Models/rf.pkl`
- **Decision Tree**: ~88% test accuracy with max_depth=85
- **Logistic Regression**: Baseline model (~74% test accuracy)
- **TF-IDF Vectorizer**: Saved as `Models/tfidf.pkl` for consistent inference
- **Tracked Models**: Logistic Regression and Random Forest versioned with DVC (`models.dvc`)

## Technologies Used

- **Python 3.11**
- **Deep Learning**: TensorFlow/Keras
- **ML Libraries**: scikit-learn
- **NLP**: NLTK
- **API**: FastAPI, Uvicorn, Pydantic
- **Version Control**: DVC (Data Version Control)
- **Experiment Tracking**: MLflow
- **Data Handling**: pandas, numpy
- **Model Persistence**: joblib

## API — FastAPI Deployment

The project exposes both the Random Forest (traditional ML) and LSTM (deep learning) models through a single FastAPI application (`main.py`). All models and artifacts are loaded once at startup via a lifespan context manager. The LSTM's `TextVectorization` vocabulary is restored from `Models/TextVectorVocab.pkl` at startup using `set_vocabulary()`.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/root` | Welcome message |
| GET | `/health` | Check if models are loaded and ready |
| POST | `/predict/trad-ml` | Predict sentiment using Random Forest (with full preprocessing pipeline) |
| POST | `/predict/dl` | Predict sentiment using LSTM |

### Request & Response

Both prediction endpoints share the same schema:

**Request body:**
```json
{ "tweet": "I absolutely love this game!" }
```

**Response:**
```json
{
  "sentiment": "Positive",
  "probability": 0.94
}
```

Sentiment values: `Positive`, `Neutral`, `Negative`, `Irrelevant`

The `/predict/trad-ml` endpoint applies the full NLP preprocessing pipeline (tokenization, lowercasing, stopword removal, lemmatization) before TF-IDF transformation, while `/predict/dl` passes the raw tweet directly to the LSTM with the restored vectorization layer.

### Running the API

```bash
# Install API dependencies
pip install fastapi uvicorn

# Start the server
uvicorn main:app --reload
```

Interactive docs available at `http://127.0.0.1:8000/docs`.

## Installation

```bash
# Clone the repository
git clone https://github.com/KaranMatt/Twitter-Sentiment-Analysis
cd Twitter-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt
```

Download NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

Pull DVC-tracked files:
```bash
dvc pull data.dvc
dvc pull models.dvc
```

## Usage

**Via the API (recommended):**

```bash
# Traditional ML endpoint
curl -X POST "http://127.0.0.1:8000/predict/trad-ml" \
     -H "Content-Type: application/json" \
     -d '{"tweet": "I absolutely love this game!"}'

# Deep Learning endpoint
curl -X POST "http://127.0.0.1:8000/predict/dl" \
     -H "Content-Type: application/json" \
     -d '{"tweet": "I absolutely love this game!"}'
```

**Directly in Python:**

```python
# Train Deep Learning Model
model.fit(train_inputs, train_target,
          epochs=20,
          validation_data=(val_inputs, val_target),
          callbacks=[early_stopping, reduce_lr])

# Save vocab for deployment
import joblib
vocab = text_vectorizer.get_vocabulary()
joblib.dump(vocab, 'Models/TextVectorVocab.pkl')

# Train Traditional ML Model
model_forest = RandomForestClassifier(n_estimators=120, max_depth=250)
model_forest.fit(train_tfidf, train_label)
```

**Version Control with DVC:**
```bash
dvc add Data/twitter_training.csv
dvc add Models/rf.pkl
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

## Experiment Tracking

All experiments are tracked using MLflow, including:
- Model hyperparameters
- Training/validation metrics
- Model artifacts
- Tags for easy filtering
- Comparison across traditional and deep learning approaches

## Version Control Strategy

### DVC Tracked Assets
- **data.dvc**: Points to `twitter_training.csv` and `twitter_validation.csv`
- **models.dvc**: Points to versioned models:
  - Logistic Regression baseline (`logreg_model.pkl`)
  - TF-IDF vectorizer (`tfidf.pkl`)
  - Random Forest — best performer (`rf.pkl`)
  - Two best LSTM variants (`lstm.h5` / `lstm.keras`)
  - TextVectorization vocabulary (`TextVectorVocab.pkl`)

### Benefits of DVC Integration
- Reproducible experiments across team members
- Version control for large data files and models
- Easy rollback to previous model versions
- Efficient storage with remote backends
- Seamless collaboration without storing large binaries in Git

## Project Structure

```
Twitter-Sentiment-Analysis/
│
├── Data/
│   ├── twitter_training.csv       # DVC-tracked (git-ignored)
│   └── twitter_validation.csv    # DVC-tracked (git-ignored)
│
├── Models/
│   ├── lstm.keras                 # DVC-tracked LSTM model (git-ignored)
│   ├── rf.pkl                     # DVC-tracked Random Forest (git-ignored)
│   ├── tfidf.pkl                  # DVC-tracked TF-IDF vectorizer (git-ignored)
│   └── TextVectorVocab.pkl        # DVC-tracked TextVectorization vocab (git-ignored)
│
├── data.dvc                       # DVC pointer for dataset
├── models.dvc                     # DVC pointer for models
├── mlruns/                        # MLflow experiment logs (git-ignored)
├── Twitter-Sentimental.ipynb      # Main notebook
├── main.py                        # FastAPI application
├── .gitignore                     # Excludes Models/, Data/, mlruns/, __pycache__/
├── .dvc/                          # DVC configuration
├── .dvcignore
├── requirements.txt
└── README.md
```

## Key Insights

- **Traditional models outperform deep learning** on this moderately-sized dataset due to limited complexity and insufficient data for deep models to capture intricate sequential patterns
- Text preprocessing (lemmatization, stopword removal) significantly improves performance
- Deep learning models show signs of overfitting despite regularization techniques
- Class balance is relatively good, reducing need for resampling techniques
- TF-IDF + Random Forest provides an excellent baseline that's hard to beat for this task
- Saving the `TextVectorization` vocabulary separately enables reliable DL model deployment without re-adapting the layer at inference time

## Future Improvements

- [ ] Implement transfer learning with pre-trained embeddings (Word2Vec, GloVe)
- [ ] Experiment with transformer-based models (BERT, RoBERTa) for larger datasets
- [ ] Add cross-validation for more robust evaluation
- [x] Deploy model as REST API using FastAPI
- [ ] Implement real-time tweet classification
- [ ] Expand dataset size to better leverage deep learning capabilities
- [ ] Create DVC pipelines for automated retraining
- [ ] Create web interface with Streamlit/Gradio

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
fastapi
uvicorn
pydantic
```

## License

This project is open-source and available for educational purposes.

## Author

Karan Mattoo

GitHub: [KaranMatt/Twitter-Sentiment-Analysis](https://github.com/KaranMatt/Twitter-Sentiment-Analysis)

---
