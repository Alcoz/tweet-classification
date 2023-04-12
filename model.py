import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 2048, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(2048, 512, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(512, output_dim, dtype=torch.float64),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

tk = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_clean(text):
    tokens = tk.tokenize(text)
    
    # lower text
    tokens = [stringliteral.lower() for stringliteral in tokens]
    
    # remove urls
    tokens = [re.sub(r'http\S+', '', stringliteral) for stringliteral in tokens]
    
    # remove @user
    tokens = [re.sub(r'@\S+', '', stringliteral) for stringliteral in tokens]
    
    # remove number
    tokens = [re.sub(r'[0-9]*', '', stringliteral) for stringliteral in tokens]
    
    tokens = [lemmatizer.lemmatize(stringliteral) for stringliteral in tokens]
    
    # remove stop_words
    tokens = list(filter(lambda word: word not in stop_words, tokens))
    
    # remove None 
    tokens = list(filter(None, tokens))
    
    return " ".join(tokens)

df_train = pd.read_csv("data/Corona_NLP_train.csv", encoding = "ISO-8859-1")
df_test = pd.read_csv("data/Corona_NLP_test.csv", encoding = "ISO-8859-1")

df_train = df_train[["OriginalTweet", "Sentiment"]]
df_test = df_test[["OriginalTweet", "Sentiment"]]

df_train["OriginalTweet"] = df_train["OriginalTweet"].apply(text_clean)
df_test["OriginalTweet"] = df_test["OriginalTweet"].apply(text_clean)

df = pd.concat([df_train, df_test])

vectorizer = TfidfVectorizer(max_features = 2500)
vectorizer.fit(df["OriginalTweet"])

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def vec(tweet):
    vectorized_tweet = vectorizer.transform([tweet])
    return torch.Tensor(vectorized_tweet.toarray()).to(torch.float64).to(device)

    