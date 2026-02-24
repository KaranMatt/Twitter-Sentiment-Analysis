import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Literal
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from contextlib import asynccontextmanager
import joblib
import tensorflow as tf


def preprocess_text(text:str)->str:
    tweet_tokens=TweetTokenizer()
    tokens=tweet_tokens.tokenize(text)
    lower_tokens=[token.lower() for token in tokens]
    stop_words=set(stopwords.words('english'))
    no_stopwords_tokens=[token for token in lower_tokens if token not in stop_words]
    lemmatizer=WordNetLemmatizer()
    lemmatized_tokens=[lemmatizer.lemmatize(token,pos='v') for token in no_stopwords_tokens]
    cleaned_tokens=[token for token in lemmatized_tokens if token.isalnum()]
    processed_text=' '.join(cleaned_tokens)
    return processed_text

class TweetInput(BaseModel):
    tweet:str=Field(min_length=1)

class SentimentResponse(BaseModel):
    sentiment:Literal['Positive','Neutral','Negative','Irrelevant']
    probability:float

tfidf=None
model_rf=None
model_lstm=None
vocab=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global tfidf,model_rf,model_lstm,vocab

    tfidf=joblib.load('Models/tfidf.pkl')
    print('Tfidf Loaded')
    model_rf=joblib.load('Models/rf.pkl')
    print('Random Forest Model Loaded')
    model_lstm=tf.keras.models.load_model('Models/lstm.h5',compile=False)
    print('DL Model Loaded')
    vocab=joblib.load('Models/TextVectorVocab.pkl')
    print('Vocab dl Loaded')
    model_lstm.layers[1].set_vocabulary(vocab)

    yield

    print('Shutdown')
    tfidf=None
    model_rf=None
    model_lstm=None
    vocab=None

app=FastAPI(title='Twitter Sentiment API',lifespan=lifespan)

@app.get('/root')
def root():
    return {'message':'Welcome to the Twitter API'}

@app.get('/health')
def health():
    if tfidf:
        return {'status':'active'}
    else:
        return {'status':'Not Active'}
    
@app.post('/predict/trad-ml',response_model=SentimentResponse)
def predict(text:TweetInput):
    preprocessed_text=preprocess_text(text=text.tweet)
    tfidf_features=tfidf.transform([preprocessed_text])
    prediction=model_rf.predict(tfidf_features)[0]
    prob=model_rf.predict_proba(tfidf_features)[0].max()
    
    if prediction==0:
        sentiment='Positive'
    elif prediction==1:
        sentiment='Neutral'
    elif prediction==2:
        sentiment='Negative'
    else:
        sentiment='Irrelevant'            
    
    return SentimentResponse(sentiment=sentiment,probability=prob)

@app.post('/predict/dl',response_model=SentimentResponse)
def predict_dl(text:TweetInput):
    string_list=tf.constant([[text.tweet]])
    predictions=model_lstm(string_list,verbose=0,training=False).numpy()[0]
    pred_class=int(np.argmax(predictions))
    prob=float(predictions.max())

    if pred_class==0:
        sentiment='Positive'
    elif pred_class==1:
        sentiment='Neutral'
    elif pred_class==2:
        sentiment='Negative'
    else:
        sentiment='Irrelevant'            
    
    return SentimentResponse(sentiment=sentiment,probability=prob)




