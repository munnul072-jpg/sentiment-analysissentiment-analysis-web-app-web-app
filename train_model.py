"""
Model Training Script for Sentiment Analysis
B.Tech Major Project - KIET Group of Institutions
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK components
def download_nltk_resources():
    """Download necessary NLTK data files"""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()

# Initialize text processing tools
stemmer_engine = PorterStemmer()
stopword_list = set(stopwords.words('english'))

def apply_text_normalization(raw_content):
    """
    Comprehensive text preprocessing pipeline
    """
    # Convert to uniform case
    normalized_text = str(raw_content).lower()
    
    # Remove URLs, mentions, and special characters
    normalized_text = re.sub(r'http\S+|www\S+|@\S+', '', normalized_text)
    normalized_text = re.sub(r'[^a-zA-Z\s]', '', normalized_text)
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
    
    # Tokenization
    try:
        token_sequence = word_tokenize(normalized_text)
    except:
        token_sequence = normalized_text.split()
    
    # Apply stopword removal and stemming
    processed_tokens = []
    for token in token_sequence:
        if token not in stopword_list and len(token) > 2:
            stemmed_token = stemmer_engine.stem(token)
            processed_tokens.append(stemmed_token)
    
    return " ".join(processed_tokens)

def load_training_corpus(data_path='dataset/train.csv'):
    """
    Load and prepare the sentiment analysis dataset
    """
    print("\n" + "="*70)
    print("STEP 1: DATASET LOADING AND PREPARATION")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}")
        print("[INFO] Creating sample dataset for testing...")
        
        # Create sample dataset
        sample_data = []
        positive_tweets = [
            "I love this product! It's amazing!",
            "Great experience, highly recommended",
            "Wonderful service, very satisfied",
            "Excellent quality, best purchase ever",
            "Fantastic, I'm so happy with this"
        ]
        negative_tweets = [
            "Terrible product, waste of money",
            "Very disappointed, not worth it",
            "Bad experience, would not recommend",
            "Poor quality, very unhappy",
            "Worst purchase ever, regret buying"
        ]
        
        for i in range(1000):
            for tweet in positive_tweets:
                sample_data.append([tweet + f" {i}", 'Positive'])
            for tweet in negative_tweets:
                sample_data.append([tweet + f" {i}", 'Negative'])
        
        data_frame = pd.DataFrame(sample_data, columns=['text', 'sentiment'])
        os.makedirs('dataset', exist_ok=True)
        data_frame.to_csv(data_path, index=False)
        print(f"[INFO] Sample dataset created with {len(data_frame)} samples")
    else:
        # Attempt to load with different encodings
        try:
            data_frame = pd.read_csv(data_path, encoding='latin-1')
            print(f"[SUCCESS] Dataset loaded from: {data_path}")
        except UnicodeDecodeError:
            data_frame = pd.read_csv(data_path, encoding='ISO-8859-1')
            print(f"[SUCCESS] Dataset loaded with ISO-8859-1 encoding")
        except Exception as e:
            print(f"[ERROR] Could not load dataset: {e}")
            raise
    
    print(f"[INFO] Dataset dimensions: {data_frame.shape[0]} rows × {data_frame.shape[1]} columns")
    
    # Detect dataset format
    if 'target' in data_frame.columns and 'text' in data_frame.columns:
        working_data = data_frame[['text', 'target']].copy()
        working_data.columns = ['content', 'polarity']
        working_data['polarity'] = working_data['polarity'].map({0: 'Negative', 4: 'Positive'})
        print("[INFO] Detected Sentiment140 dataset format")
        
    elif 'sentiment' in data_frame.columns and 'text' in data_frame.columns:
        working_data = data_frame[['text', 'sentiment']].copy()
        working_data.columns = ['content', 'polarity']
        print("[INFO] Detected standard dataset format")
        
    else:
        working_data = data_frame.iloc[:, :2].copy()
        working_data.columns = ['content', 'polarity']
        print("[INFO] Using auto-detected column mapping")
    
    # Remove missing values
    initial_count = len(working_data)
    working_data = working_data.dropna()
    print(f"[INFO] Removed {initial_count - len(working_data)} missing entries")
    
    # Standardize sentiment labels
    working_data['polarity'] = working_data['polarity'].astype(str).str.capitalize()
    valid_categories = ['Positive', 'Negative']
    working_data = working_data[working_data['polarity'].isin(valid_categories)]
    
    # Display class distribution
    print("\n[INFO] Class Distribution:")
    class_counts = working_data['polarity'].value_counts()
    for class_name, count in class_counts.items():
        percentage = (count / len(working_data)) * 100
        print(f"   {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    return working_data['content'], working_data['polarity']

def extract_text_features(text_sequences, max_feature_count=5000):
    """
    Convert text data to numerical features using TF-IDF vectorization
    """
    print("\n" + "="*70)
    print("STEP 2: FEATURE EXTRACTION (TF-IDF VECTORIZATION)")
    print("="*70)
    
    feature_extractor = TfidfVectorizer(
        max_features=max_feature_count,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )
    
    print(f"[INFO] Generating {max_feature_count} features using TF-IDF")
    feature_matrix = feature_extractor.fit_transform(text_sequences)
    print(f"[SUCCESS] Feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix, feature_extractor

def evaluate_classifier(model_instance, model_name, training_features, test_features, 
                       training_labels, test_labels):
    """
    Comprehensive evaluation of a single classifier
    """
    print(f"\n[INFO] Training {model_name}...")
    model_instance.fit(training_features, training_labels)
    
    predicted_labels = model_instance.predict(test_features)
    
    accuracy_value = accuracy_score(test_labels, predicted_labels) * 100
    precision_value = precision_score(test_labels, predicted_labels, average='weighted') * 100
    recall_value = recall_score(test_labels, predicted_labels, average='weighted') * 100
    f1_value = f1_score(test_labels, predicted_labels, average='weighted') * 100
    
    print(f"[SUCCESS] {model_name} Performance:")
    print(f"   → Accuracy:  {accuracy_value:.2f}%")
    print(f"   → Precision: {precision_value:.2f}%")
    print(f"   → Recall:    {recall_value:.2f}%")
    print(f"   → F1-Score:  {f1_value:.2f}%")
    
    return {
        'model': model_name,
        'accuracy': accuracy_value,
        'precision': precision_value,
        'recall': recall_value,
        'f1_score': f1_value,
        'predictions': predicted_labels,
        'trained_model': model_instance
    }

def generate_confusion_matrix_visualization(true_labels, predicted_labels, classifier_name, 
                                           save_directory='results'):
    """
    Create and save confusion matrix visualization (without displaying)
    """
    os.makedirs(save_directory, exist_ok=True)
    
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    class_categories = ['Negative', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=class_categories, yticklabels=class_categories,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title(f'Confusion Matrix - {classifier_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('Actual Category', fontsize=12)
    plt.tight_layout()
    
    filename = f"{save_directory}/confusion_matrix_{classifier_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # IMPORTANT: Close figure to free memory
    print(f"[INFO] Confusion matrix saved: {filename}")
    return confusion_mat

def execute_training_pipeline():
    """
    Main training pipeline orchestrator
    """
    print("\n" + "="*70)
    print("MULTILINGUAL SENTIMENT ANALYSIS TRAINING PIPELINE")
    print("B.Tech Major Project 2025-26")
    print("="*70)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load and prepare dataset
    text_data, label_data = load_training_corpus()
    
    # Apply text preprocessing
    print("\n" + "="*70)
    print("STEP 3: TEXT PREPROCESSING")
    print("="*70)
    print("[INFO] Applying normalization pipeline...")
    
    processed_texts = []
    total_samples = len(text_data)
    for idx, text in enumerate(text_data):
        if idx % 1000 == 0:
            print(f"   Processing sample {idx:,} of {total_samples:,}")
        processed_texts.append(apply_text_normalization(text))
    
    print(f"[SUCCESS] Preprocessed {len(processed_texts):,} text samples")
    
    # Extract features
    feature_matrix, vectorizer_model = extract_text_features(processed_texts)
    
    # Split data
    print("\n" + "="*70)
    print("STEP 4: DATA PARTITIONING")
    print("="*70)
    
    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        feature_matrix, label_data, test_size=0.10, random_state=42, stratify=label_data
    )
    
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features, train_val_labels, test_size=0.20, random_state=42, stratify=train_val_labels
    )
    
    print(f"[INFO] Training set:   {train_features.shape[0]:,} samples")
    print(f"[INFO] Validation set: {val_features.shape[0]:,} samples")
    print(f"[INFO] Testing set:    {test_features.shape[0]:,} samples")
    
    # Define classifiers
    classification_models = {
        'Logistic Regression (LR)': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Support Vector Machine (SVM)': SVC(kernel='linear', random_state=42, C=1.0, probability=True),
        'Random Forest (RF)': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5, weights='distance')
    }
    
    # Train and evaluate
    print("\n" + "="*70)
    print("STEP 5: MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    evaluation_results = []
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    for model_label, model_obj in classification_models.items():
        try:
            result = evaluate_classifier(
                model_obj, model_label,
                train_features, test_features,
                train_labels, test_labels
            )
            evaluation_results.append(result)
            
            generate_confusion_matrix_visualization(test_labels, result['predictions'], model_label)
            
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = result['trained_model']
                best_model_name = model_label
        except Exception as e:
            print(f"[ERROR] Failed to train {model_label}: {e}")
    
    # Display results
    print("\n" + "="*70)
    print("STEP 6: PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    print("\nTable 1: Comparative Analysis of Classification Models")
    print("-"*70)
    print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for result in evaluation_results:
        print(f"{result['model']:<35} {result['accuracy']:<12.2f} {result['precision']:<12.2f} "
              f"{result['recall']:<12.2f} {result['f1_score']:<12.2f}")
    
    if best_model_name:
        print("-"*70)
        print(f"\n[SUCCESS] Best performing model: {best_model_name}")
        print(f"[SUCCESS] Highest accuracy achieved: {best_accuracy:.2f}%")
        
        # Save models
        print("\n" + "="*70)
        print("STEP 7: MODEL PERSISTENCE")
        print("="*70)
        
        joblib.dump(best_model, 'models/sentiment_model.pkl')
        joblib.dump(vectorizer_model, 'models/vectorizer.pkl')
        
        print("[SUCCESS] Best model saved to: models/sentiment_model.pkl")
        print("[SUCCESS] Feature vectorizer saved to: models/vectorizer.pkl")
        
        # Save results to CSV
        results_dataframe = pd.DataFrame([
            {
                'Model': r['model'],
                'Accuracy (%)': round(r['accuracy'], 2),
                'Precision (%)': round(r['precision'], 2),
                'Recall (%)': round(r['recall'], 2),
                'F1-Score (%)': round(r['f1_score'], 2)
            }
            for r in evaluation_results
        ])
        
        results_dataframe.to_csv('results/model_performance_comparison.csv', index=False)
        print("[SUCCESS] Performance metrics saved to: results/model_performance_comparison.csv")
    else:
        print("\n[ERROR] No model trained successfully!")
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED")
    print("="*70)
    
    return evaluation_results, best_model, vectorizer_model

if __name__ == "__main__":
    execute_training_pipeline()