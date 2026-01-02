import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the dataset
dataset_path = '../data/raw/spam_dataset_cleaned.csv'
df = pd.read_csv(dataset_path, encoding='latin1')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove rows with missing values in clean_text or label_str
df = df.dropna(subset=['clean_text', 'label_str'])

# Convert clean_text to string and remove any remaining NaN
df['clean_text'] = df['clean_text'].astype(str)

# Remove rows where clean_text is 'nan' string or empty
df = df[df['clean_text'].str.strip() != '']
df = df[df['clean_text'] != 'nan']

print(f"\nDataset shape after cleaning: {df.shape}")
print("\nClass Distribution:")
print(df['label_str'].value_counts())

# Text Preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Apply transformation to the dataset
print("\nApplying text transformation to dataset...")
df['transformed_text'] = df['clean_text'].apply(transform_text)

# Prepare features and target
X = df['transformed_text']  # Use transformed text instead of clean_text
y = df['label_str']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Convert text to TF-IDF features
print("\nConverting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=3000)  # Removed stop_words since transform_text already handles it
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Train SVC model
print("\nTraining SVC model...")
svc_model = SVC(kernel='linear', C=1.0, random_state=42, probability=True)  # Added probability=True
svc_model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# Make predictions
y_pred = svc_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save the model and vectorizer
with open('../data/processed/svc_model.pkl', 'wb') as f:
    pickle.dump(svc_model, f)

with open('../data/processed/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")

# Function to predict new messages (FIXED)
def predict_message(message):
    """
    Predict if a message is spam or ham
    """
    # Transform the message first
    transformed_message = transform_text(message)
    # Then vectorize
    message_tfidf = vectorizer.transform([transformed_message])
    prediction = svc_model.predict(message_tfidf)[0]
    return prediction

# Enhanced function with confidence scores (NEW FEATURE)
def predict_message_with_confidence(message):
    """
    Predict if a message is spam or ham with confidence scores
    """
    transformed_message = transform_text(message)
    message_tfidf = vectorizer.transform([transformed_message])
    prediction = svc_model.predict(message_tfidf)[0]
    probabilities = svc_model.predict_proba(message_tfidf)[0]
    
    confidence = {
        'prediction': prediction,
        'spam_probability': probabilities[1] if len(probabilities) > 1 else probabilities[0],
        'ham_probability': probabilities[0] if len(probabilities) > 1 else probabilities[0]
    }
    return confidence

# Batch prediction function (NEW FEATURE)
def predict_batch(messages):
    """
    Predict multiple messages at once
    """
    results = []
    for msg in messages:
        transformed = transform_text(msg)
        message_tfidf = vectorizer.transform([transformed])
        prediction = svc_model.predict(message_tfidf)[0]
        results.append({
            'message': msg,
            'transformed': transformed,
            'prediction': prediction
        })
    return pd.DataFrame(results)

# Test with some examples
print(f"\n{'='*50}")
print("TESTING THE MODEL")
print(f"{'='*50}")

test_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account will be closed. Verify your details immediately.",
    "Can you send me the report by end of day?",
]

for msg in test_messages:
    prediction = predict_message(msg)
    print(f"\nMessage: {msg}")
    print(f"Prediction: {prediction.upper()}")
    print("-" * 50)

# Test with confidence scores (NEW FEATURE)
print(f"\n{'='*50}")
print("TESTING WITH CONFIDENCE SCORES")
print(f"{'='*50}")

for msg in test_messages:
    result = predict_message_with_confidence(msg)
    print(f"\nMessage: {msg}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Spam Probability: {result['spam_probability']:.2%}")
    print(f"Ham Probability: {result['ham_probability']:.2%}")
    print("-" * 50)

# Batch prediction example (NEW FEATURE)
print(f"\n{'='*50}")
print("BATCH PREDICTION")
print(f"{'='*50}")

batch_results = predict_batch(test_messages)
print("\nBatch Results:")
print(batch_results[['message', 'prediction']].to_string(index=False))

print("\n✓ Spam detection model is ready to use!")
print("✓ Model saved as 'svc_model.pkl'")
print("✓ Vectorizer saved as 'vectorizer.pkl'")
print("\n🆕 NEW FEATURES ADDED:")
print("  • predict_message_with_confidence() - Get prediction with probabilities")
print("  • predict_batch() - Process multiple messages at once")
print("  • Automatic NLTK data download")
print("  • Fixed transform_text integration in predictions")