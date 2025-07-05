# %%
# Configure GPU settings at the start
import tensorflow as tf

# Enable GPU memory growth to prevent OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

print("Num GPUs Available:", len(gpus))
print("TF Version:", tf.__version__)
print("Physical Devices:", tf.config.list_physical_devices())

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import joblib
import matplotlib.pyplot as plt

# Mixed precision for faster training (requires compatible GPU)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# %%
try:
    df = pd.read_csv('phishing_site_urls_updated.csv')
    print(f"Data loaded: {df.shape[0]} rows")
except Exception as e:
    print(f"Failed to read CSV: {e}")
    raise

# %%
# Clean URLs
df['URL'] = (
    df['URL']
    .str.replace(r'^https?://', '', regex=True)
    .str.replace(r'^www\.', '', regex=True)
    .str.rstrip('/')
)

# %%
url_lengths = df['URL'].apply(len)
print("URL Length Stats:")
print(url_lengths.describe())
plt.hist(url_lengths, bins=50)
plt.title("URL Length Distribution")
plt.show()

# %%
# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    df['URL'], df['Label'], test_size=0.2, stratify=df['Label'], random_state=42
)

print(f"Training set: {X_train.shape[0]} URLs")
print(f"Test set: {X_test.shape[0]} URLs")

# %%
# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# %%
# Character Embeddings
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# %%
# Logistic Regression Baseline
model_lr = LogisticRegression(n_jobs=-1)  # Use all CPU cores
model_lr.fit(X_train_tfidf, y_train)

# %%
# LSTM Model with GPU context
max_len = min(100, int(np.percentile(df['URL'].apply(len), 95)))
print(f"Using sequence length: {max_len}")

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Convert to numpy arrays
X_train_pad = np.array(X_train_pad)
X_test_pad = np.array(X_test_pad)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Class weights
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights = {0: class_weights[0], 1: class_weights[1]}

# Build model within GPU context
with tf.device('/GPU:0'):
    model_lstm = Sequential([
        Embedding(input_dim=len(tokenizer.word_index)+1, 
                  output_dim=256,
                  input_length=max_len,
                  mask_zero=True),
        
        Bidirectional(LSTM(128, return_sequences=True,
                          kernel_regularizer=l2(0.01))),
        Dropout(0.5),
        BatchNormalization(),
        
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(1, activation='sigmoid', dtype='float32')  # Output in float32
    ])
    
    optimizer = Adam(learning_rate=0.01)
    model_lstm.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'precision', 'recall']
    )

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# Train with increased batch size
history = model_lstm.fit(
    X_train_pad,
    y_train,
    epochs=50,
    batch_size=512,  # Increased for GPU efficiency
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# %%
# Evaluation
print("\n=== Logistic Regression ===")
y_pred = model_lr.predict(X_test_tfidf)
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, model_lr.predict_proba(X_test_tfidf)[:,1]):.4f}")

print("\n=== LSTM Model ===")
y_pred_lstm = (model_lstm.predict(X_test_pad, batch_size=512) > 0.5).astype(int)
print(f"Precision: {precision_score(y_test, y_pred_lstm):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lstm):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lstm):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, model_lstm.predict(X_test_pad, batch_size=512)):.4f}")

# %%
# Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax[0])
ax[0].set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test, y_pred_lstm), annot=True, fmt='d', ax=ax[1])
ax[1].set_title('LSTM')
plt.tight_layout()
plt.savefig('confusion_matrices.png')

# %%
# Save models
joblib.dump(model_lr, "logistic_regression_phishing.pkl")
model_lstm.save("lstm_phishing_model.h5")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(tokenizer, "tokenizer.pkl")
print("Models saved")

# %%
# Prediction functions
def clean_url(url):
    return (url.replace('http://', '')
            .replace('https://', '')
            .replace('www.', '')
            .rstrip('/'))

def predict_phishing(url):
    """Traditional model prediction"""
    url_clean = clean_url(url)
    url_tfidf = vectorizer.transform([url_clean])
    return "Phishing" if model_lr.predict(url_tfidf)[0] == 1 else "Legitimate"

def predict_phishing_lstm(url):
    """LSTM model prediction with confidence"""
    url_clean = clean_url(url)
    url_seq = tokenizer.texts_to_sequences([url_clean])
    url_padded = pad_sequences(url_seq, maxlen=max_len)
    prob = model_lstm.predict(url_padded, verbose=0)[0][0]
    return ("Phishing" if prob > 0.5 else "Legitimate"), float(prob)

# %%
# Test predictions
test_urls = [
    "google.com",
    "www.avedeoiro.com/site/plugins/chase/",
    "https://www.paypal-login.scam/verify",
    "www.agenciasports.com/wp-includes/images/smilies/mbafresh.htm"
]

print("Predictions:")
for url in test_urls:
    result, confidence = predict_phishing_lstm(url)
    print(f"{url[:60]:<60} | {result:<10} | Confidence: {confidence:.4f}")