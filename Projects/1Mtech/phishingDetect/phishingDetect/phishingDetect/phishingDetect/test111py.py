# %%
# In a Jupyter notebook cell
# !pip install pandas numpy seaborn tensorflow scikit-learn keras joblib

# %%
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

# %%
import sys
print(sys.executable)

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense , Bidirectional, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
import joblib
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# %%
import tensorflow as tf
print(tf.config.list_physical_devices())
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())

# %%
try:
    df = pd.read_csv('phishing_site_urls_updated.csv')
    print(df)
except Exception as e:
    print(f"Failed to read the CSV file: {e}")


# %% [markdown]
# Clean URLs

# %%
# 1. Data Cleaning: Remove both http(s) and www, and trailing slashes
df['URL'] = (
    df['URL']
    .str.replace(r'^https?://', '', regex=True)
    .str.replace(r'^www\.', '', regex=True)
    .str.rstrip('/')
)

# %%
url_lengths = df['URL'].apply(len)
print(url_lengths.describe())

# %% [markdown]
# Split Data

# %%
# 3. Use stratified split for balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    df['URL'], df['Label'], test_size=0.2, stratify=df['Label'], random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Training labels size: {y_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Test labels size: {y_test.shape[0]}")

# %% [markdown]
# TF-IDF (Traditional NLP)

# %%
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))  # Use character-level n-grams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# %% [markdown]
# Character Embeddings (Deep Learning)

# %%
# 2. Remove duplicate Tokenizer fitting (keep only one)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# %% [markdown]
# Model Building
# Model 1: Logistic Regression (Baseline)

# %%
model_lr = LogisticRegression()
model_lr.fit(X_train_tfidf, y_train)

# %% [markdown]
# Model 2: LSTM (Deep Learning)

# %%
# 4. Set max_len dynamically based on URL length distribution
max_len = min(100, int(np.percentile(df['URL'].apply(len), 95)))

# Pad sequences to equal length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Convert to NumPy arrays
X_train_pad = np.array(X_train_pad)
X_test_pad = np.array(X_test_pad)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Calculate class weights for imbalanced data
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: class_weights[0], 1: class_weights[1]}

# Enhanced LSTM Model Architecture
model_lstm = Sequential([
    # Increased embedding dimension with masking
    Embedding(input_dim=len(tokenizer.word_index)+1, 
                output_dim=256,
                input_length=max_len,
                mask_zero=True),
    
    # First Bidirectional LSTM with regularization
    Bidirectional(LSTM(128, return_sequences=True,
                      kernel_regularizer=l2(0.01))),
    Dropout(0.5),
    BatchNormalization(),
    
    # Second Bidirectional LSTM
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    BatchNormalization(),
    
    # Output layer
    Dense(1, activation='sigmoid')
])

# Custom optimizer with learning rate decay
optimizer = Adam(learning_rate=0.01)
model_lstm.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Train with class weights and callbacks
history = model_lstm.fit(
    X_train_pad,
    y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# %% [markdown]
# Evaluation

# %%
# For logistic regression
y_pred = model_lr.predict(X_test_tfidf)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# 6. Add ROC-AUC for both models in evaluation
y_pred_proba = model_lr.predict_proba(X_test_tfidf)[:, 1]
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

# For LSTM
y_pred_lstm = (model_lstm.predict(X_test_pad) > 0.5).astype(int)
print("LSTM F1-Score:", f1_score(y_test, y_pred_lstm))

y_pred_lstm_proba = model_lstm.predict(X_test_pad)
print("LSTM ROC-AUC:", roc_auc_score(y_test, y_pred_lstm_proba))

# %% [markdown]
# Confusion Matrix Analysis

# %%
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')

# %% [markdown]
# False Positive Analysis

# %%
false_positives = X_test[(y_pred == 1) & (y_test == 0)]
print("False Positives:", false_positives.head())

# %%
# 7. Save models for deployment
model_lr_path = "logistic_regression_phishing.pkl"
joblib.dump(model_lr, model_lr_path)
model_lstm.save("lstm_phishing_model.h5")

# 8. Improve prediction functions with HTTPS and trailing slash cleaning
def clean_url(url):
    return (
        url.replace('http://', '')
           .replace('https://', '')
           .replace('www.', '')
           .rstrip('/')
    )

def predict_phishing(url):
    url_clean = clean_url(url)
    url_tfidf = vectorizer.transform([url_clean])
    return "Phishing" if model_lr.predict(url_tfidf)[0] == 1 else "Legitimate"

# Test
# print(predict_phishing("google.com"))          # Legitimate
print(predict_phishing("www.avedeoiro.com/site/plugins/chase/"))   # Phishing

# %%

def predict_phishing_lstm(url):
    url_clean = clean_url(url)
    url_seq = tokenizer.texts_to_sequences([url_clean])
    url_padded = pad_sequences(url_seq, maxlen=max_len)
    prediction_prob = model_lstm.predict(url_padded)[0][0]
    print(f"Probability: {prediction_prob:.3f}")
    return "Phishing" if prediction_prob > 0.5 else "Legitimate"

# %%
# Example 1: Legitimate URL
print(predict_phishing_lstm("google.com"))          # Output: Legitimate

# Example 2: Phishing URL
print(predict_phishing_lstm("paypal-login.scam"))   # Output: Phishing

# Example 3: Edge Case
print(predict_phishing_lstm("www.agenciasports.com/wp-includes/images/smilies/mbafresh.htm"))  # Depends on training data


