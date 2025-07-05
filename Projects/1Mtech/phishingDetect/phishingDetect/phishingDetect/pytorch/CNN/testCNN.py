import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import re

# --- Load artifacts ---
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model_lr = joblib.load("logistic_regression_phishing.pkl")
char_vocab = joblib.load("char_vocab.pkl")
max_len = joblib.load("max_len.pkl")

# --- Rebuild CNN model ---
class PhishingCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(char_vocab)
embedding_dim = 128
n_filters = 100
filter_sizes = [3, 4, 5]
output_dim = 1
dropout = 0.5
model = PhishingCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout)
model.load_state_dict(torch.load("cnn_phishing_model.pt", map_location=device))
model.to(device)
model.eval()

# --- Helper functions ---
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

# --- Example usage ---
test_url = "https://www.paypal-login.scam/verify"
res_lr, conf_lr = predict_phishing(test_url, 'lr')
res_cnn, conf_cnn = predict_phishing(test_url, 'cnn')
print(f"URL: {test_url}")
print(f"  LR  : {res_lr} (confidence: {conf_lr:.4f})")
print(f"  CNN : {res_cnn} (confidence: {conf_cnn:.4f})")

# Interactive prediction loop
print("\nEnter URLs to check (type 'exit' to quit):")
while True:
    user_url = input("URL: ").strip()
    if user_url.lower() == 'exit':
        break
    res_lr, conf_lr = predict_phishing(user_url, 'lr')
    res_cnn, conf_cnn = predict_phishing(user_url, 'cnn')
    print(f"  [LR ] {res_lr} (confidence: {conf_lr:.4f})")
    print(f"  [CNN] {res_cnn} (confidence: {conf_cnn:.4f})")