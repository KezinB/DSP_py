import torch
import joblib
import re
import sys

# Load artifacts
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model_lr = joblib.load("logistic_regression_phishing.pkl")
char_vocab = joblib.load("char_vocab.pkl")
max_len = joblib.load("max_len.pkl")

# LSTM model definition (must match training)
import torch.nn as nn
class PhishingLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LSTM model
vocab_size = len(char_vocab)
embedding_dim = 128
hidden_dim = 64
output_dim = 1
n_layers = 2
dropout = 0.3
model = PhishingLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
model.load_state_dict(torch.load("lstm_phishing_model.pt", map_location=device))
model.to(device)
model.eval()

def clean_url(url):
    return re.sub(r'^https?://(?:www\.)?', '', str(url)).rstrip('/')

def url_to_indices(url):
    return [char_vocab.get(char, char_vocab['<UNK>']) for char in url]

def predict_phishing(url, model_type='lr'):
    cleaned = clean_url(url)
    if model_type == 'lr':
        vectorized = vectorizer.transform([cleaned])
        proba = model_lr.predict_proba(vectorized)[0][1]
        result = "Phishing" if proba > 0.5 else "Legitimate"
    else:
        indices = url_to_indices(cleaned)
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [char_vocab['<PAD>']] * (max_len - len(indices))
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = torch.sigmoid(model(tensor))
        proba = prediction.item()
        result = "Phishing" if proba > 0.5 else "Legitimate"
    return result, float(proba)

if __name__ == "__main__":
    print("Enter a URL to test (or type 'exit' to quit):")
    while True:
        url = input("URL: ").strip()
        if url.lower() == 'exit':
            break
        res_lr, conf_lr = predict_phishing(url, 'lr')
        res_lstm, conf_lstm = predict_phishing(url, 'lstm')
        print(f"\nLogistic Regression: {res_lr} (Confidence: {conf_lr:.4f})")
        print(f"LSTM Model:         {res_lstm} (Confidence: {conf_lstm:.4f})\n")