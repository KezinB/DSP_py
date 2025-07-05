import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time
from collections import defaultdict
import re

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# %%
try:
    df = pd.read_csv('phishing_site_urls_updated.csv')
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Label distribution:\n{df['Label'].value_counts(normalize=True)}")
except Exception as e:
    print(f"Failed to read CSV: {e}")
    # Create dummy data if file not found
    print("Creating dummy data for demonstration")
    data = {'URL': [
        'google.com', 'bad-site.com/scam', 'yahoo.com', 
        'phishy-site.net/login', 'github.com', 'trusted-bank.com',
        'facebook-clone.ru/login', 'amazon-payment.verify'
    ], 'Label': [0, 1, 0, 1, 0, 0, 1, 1]}
    df = pd.DataFrame(data)

# %%
# Clean URLs
def clean_url(url):
    """Standardize URL format by removing protocols and www"""
    return re.sub(r'^https?://(?:www\.)?', '', str(url)).rstrip('/')

df['URL'] = df['URL'].apply(clean_url)
print("\nSample cleaned URLs:")
print(df['URL'].head())

# %%
# Analyze URL lengths
url_lengths = df['URL'].apply(len)
print("\nURL Length Statistics:")
print(url_lengths.describe())

plt.figure(figsize=(10, 4))
plt.hist(url_lengths, bins=50, color='skyblue')
plt.title("URL Length Distribution")
plt.xlabel("Character Count")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('url_length_distribution.png') if len(df) > 10 else plt.show()

# %%
# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    df['URL'], df['Label'], test_size=0.2, stratify=df['Label'], random_state=42
)

print(f"\nData Split:")
print(f"Training set: {X_train.shape[0]} URLs")
print(f"Test set: {X_test.shape[0]} URLs")
print(f"Train class balance: {np.mean(y_train == 1):.2f} phishing")
print(f"Test class balance: {np.mean(y_test == 1):.2f} phishing")

# %%
# TF-IDF Vectorizer - CPU optimized
print("\nFitting TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

# %%
# Logistic Regression Baseline - CPU friendly
print("\nTraining Logistic Regression...")
start_time = time.time()
model_lr = LogisticRegression(n_jobs=-1, max_iter=1000)
model_lr.fit(X_train_tfidf, y_train)
lr_time = time.time() - start_time
print(f"Logistic Regression trained in {lr_time:.1f} seconds")

# %%
# Character-level tokenizer for PyTorch
print("\nCreating character vocabulary...")
char_vocab = defaultdict(lambda: len(char_vocab))
char_vocab['<PAD>'] = 0  # Padding token
char_vocab['<UNK>'] = 1  # Unknown token

# Build vocabulary
for url in X_train:
    for char in url:
        char_vocab[char]  # This will add new characters

# Reverse mapping for decoding
idx_to_char = {idx: char for char, idx in char_vocab.items()}

# Calculate max length based on 95th percentile
max_len = min(100, int(np.percentile([len(url) for url in X_train], 95)))
print(f"Using sequence length: {max_len}")
print(f"Vocabulary size: {len(char_vocab)}")

# %%
# Convert URLs to indices
def url_to_indices(url):
    """Convert URL to list of character indices"""
    return [char_vocab.get(char, char_vocab['<UNK>']) for char in url]

# Create PyTorch datasets
class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len):
        self.urls = urls
        self.labels = labels
        self.max_len = max_len
        
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = self.urls.iloc[idx]
        indices = url_to_indices(url)
        
        # Pad or truncate sequence
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [char_vocab['<PAD>']] * (self.max_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels.iloc[idx], dtype=torch.float)

# Create datasets
train_dataset = URLDataset(X_train, y_train, max_len)
test_dataset = URLDataset(X_test, y_test, max_len)

# Create data loaders
batch_size = 256 if device.type == "cuda" else 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %%
# PyTorch CNN Model (Character-Level)
class PhishingCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                      out_channels=n_filters, 
                      kernel_size=fs) 
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        # Apply convolutions and pooling
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate features from different filter sizes
        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch_size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)

# Model parameters
vocab_size = len(char_vocab)
embedding_dim = 128
n_filters = 100
filter_sizes = [3, 4, 5]  # Detect 3,4,5 character patterns
output_dim = 1
dropout = 0.5

model = PhishingCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout)
model.to(device)

print("\nCNN Model architecture:")
print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %%
# Training configuration
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.2, patience=2, verbose=True
)

# %%
# Training function
def train(model, iterator, optimizer, criterion, scheduler=None):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(inputs).squeeze(1)
        loss = criterion(predictions, labels)
        
        # Calculate accuracy
        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == labels).float()
        acc = correct.sum() / len(correct)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    if scheduler:
        scheduler.step(epoch_loss)
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in iterator:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            predictions = model(inputs).squeeze(1)
            loss = criterion(predictions, labels)
            
            # Calculate accuracy
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == labels).float()
            acc = correct.sum() / len(correct)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            # Store for metrics calculation
            all_preds.extend(rounded_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), all_preds, all_labels

# %%
# Train the model
n_epochs = 30 if device.type == "cuda" else 10
print(f"\nTraining CNN ({device.type.upper()} mode) for {n_epochs} epochs...")

train_start = time.time()
for epoch in range(n_epochs):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc, valid_preds, valid_labels = evaluate(model, test_loader, criterion)
    
    # Calculate additional metrics
    valid_precision = precision_score(valid_labels, valid_preds, zero_division=0)
    valid_recall = recall_score(valid_labels, valid_preds, zero_division=0)
    valid_f1 = f1_score(valid_labels, valid_preds, zero_division=0)
    
    epoch_time = time.time() - start_time
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_time:.2f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
    print(f'\tPrecision: {valid_precision:.3f} | Recall: {valid_recall:.3f} | F1: {valid_f1:.3f}')

total_time = time.time() - train_start
print(f"Training completed in {total_time:.1f} seconds")

# %%
# Final evaluation on test set
test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
test_precision = precision_score(test_labels, test_preds)
test_recall = recall_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)

print("\n=== Final Test Evaluation ===")
print(f"Accuracy:  {test_acc*100:.2f}%")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

# ROC AUC (need probabilities)
model.eval()
all_probs = []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        predictions = torch.sigmoid(model(inputs).squeeze(1))
        all_probs.extend(predictions.cpu().numpy())

test_auc = roc_auc_score(test_labels, all_probs)
print(f"ROC-AUC:   {test_auc:.4f}")

# %%
# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Phishing'], 
            yticklabels=['Legitimate', 'Phishing'])
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('cnn_confusion_matrix.png')
plt.show()

# %%
# Save models
print("\nSaving models...")
joblib.dump(model_lr, "logistic_regression_phishing.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save PyTorch model and vocabulary
torch.save(model.state_dict(), "cnn_phishing_model.pt")
joblib.dump(dict(char_vocab), "char_vocab.pkl")
joblib.dump(max_len, "max_len.pkl")
print("Models and artifacts saved")

# %%
# Prediction functions
def predict_phishing(url, model_type='lr'):
    """Predict if URL is phishing with either model"""
    cleaned = clean_url(url)
    
    if model_type == 'lr':
        vectorized = vectorizer.transform([cleaned])
        proba = model_lr.predict_proba(vectorized)[0][1]
        result = "Phishing" if proba > 0.5 else "Legitimate"
    else:  # CNN
        # Convert to indices
        indices = url_to_indices(cleaned)
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [char_vocab['<PAD>']] * (max_len - len(indices))
            
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            prediction = torch.sigmoid(model(tensor))
        
        proba = prediction.item()
        result = "Phishing" if proba > 0.5 else "Legitimate"
    
    return result, float(proba)

# %%
# Test predictions
test_urls = [
    "google.com",
    "https://www.paypal-login.scam/verify",
    "http://www.legit-bank.com/login",
    "facebook.com.fake.login.page.ru",
    "github.com",
    "amazon-payment-confirm.xyz",
    "twitter.com.secure.login.id12345.com"
]

print("\nTest Predictions:")
print("-" * 70)
print(f"{'URL':<45} | {'Model':<6} | {'Result':<10} | Confidence")
print("-" * 70)
for url in test_urls:
    # Shorten URL for display
    display_url = url[:40] + "..." if len(url) > 43 else url.ljust(43)
    
    # LR prediction
    res_lr, conf_lr = predict_phishing(url, 'lr')
    print(f"{display_url:<45} | {'LR':<6} | {res_lr:<10} | {conf_lr:.4f}")
    
    # CNN prediction
    res_cnn, conf_cnn = predict_phishing(url, 'cnn')
    print(f"{display_url:<45} | {'CNN':<6} | {res_cnn:<10} | {conf_cnn:.4f}")
    print("-" * 70)