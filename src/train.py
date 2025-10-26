import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd

# Initialize MLflow
mlflow.set_experiment("Sentiment_Analysis")

df = pd.read_csv('Reviews.csv')
print(df.head())
print(df.info())

# ---- Next Cell ----

import string

df['Text'] = df['Text'].fillna('')
df['Cleaned_Text'] = df['Text'].str.lower()
df['Cleaned_Text'] = df['Cleaned_Text'].str.translate(str.maketrans('', '', string.punctuation))

# ---- Next Cell ----

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
df['Cleaned_Text'] = df['Cleaned_Text'].apply(word_tokenize)
df['Cleaned_Text'] = df['Cleaned_Text'].apply(lambda x: [word for word in x if word not in stop_words])

# ---- Next Cell ----

df['Cleaned_Text'] = df['Cleaned_Text'].apply(lambda x: ' '.join(x))
# Notebook `display()` is not available in plain python scripts — use print instead
print(df[['Text', 'Cleaned_Text']].head().to_string())

# ---- Next Cell ----

X = df['Cleaned_Text']
df['Sentiment'] = df['Score'].apply(lambda score: 'positive' if score in [4, 5] else ('negative' if score in [1, 2] else None))
df_sentiment = df.dropna(subset=['Sentiment'])
y = df_sentiment['Sentiment']
X = df_sentiment['Cleaned_Text']

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
# Replace notebook display with console output
print(df_sentiment[['Score', 'Sentiment']].head().to_string())

# ---- Next Cell ----

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
print("Shape of y_val:", y_val.shape)

# ---- Next Cell ----

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Start MLflow run for Naive Bayes
with mlflow.start_run(run_name="Naive_Bayes_Baseline"):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Log parameters
    mlflow.log_param("model_type", "MultinomialNB")
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_negative", report['negative']['precision'])
    mlflow.log_metric("recall_negative", report['negative']['recall'])
    mlflow.log_metric("f1_negative", report['negative']['f1-score'])
    mlflow.log_metric("precision_positive", report['positive']['precision'])
    mlflow.log_metric("recall_positive", report['positive']['recall'])
    mlflow.log_metric("f1_positive", report['positive']['f1-score'])
    
    # Log artifacts
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
    
    # Log model with signature
    signature = infer_signature(X_train_tfidf, model.predict(X_train_tfidf))
    mlflow.sklearn.log_model(model, "naive_bayes_model", signature=signature)
    
    # Log vectorizer
    mlflow.sklearn.log_model(vectorizer, "tfidf_vectorizer")

# ---- Next Cell ----

X_val_tfidf = vectorizer.transform(X_val)
y_pred_val = model.predict(X_val_tfidf)

accuracy_val = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {accuracy_val:.4f}")

report_val = classification_report(y_val, y_pred_val)
print("Validation Classification Report:\n", report_val)

# ---- Next Cell ----

print("Test Set Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

print("\nValidation Set Performance:")
print(f"Validation Accuracy: {accuracy_val:.4f}")
print("Validation Classification Report:\n", report_val)

# ---- Next Cell ----

# Identify the original DataFrame indices of negative reviews in the test set
negative_original_indices_in_test = y_test[y_test == 'negative'].index

# Get the indices of X_test that correspond to these original indices
negative_relative_indices_in_test = X_test.index.get_indexer(negative_original_indices_in_test)

# Get the predicted sentiments for these negative reviews using the correct indices for the sparse matrix
predicted_sentiment_for_negatives = model.predict(X_test_tfidf[negative_relative_indices_in_test])

# Find the indices within the 'predicted_sentiment_for_negatives' array where the prediction was 'positive'
misclassified_negative_relative_indices = [i for i, pred in enumerate(predicted_sentiment_for_negatives) if pred == 'positive']

# Get the original DataFrame indices of the misclassified negative reviews
misclassified_negative_original_indices = negative_original_indices_in_test[misclassified_negative_relative_indices]

# Display some of the misclassified negative reviews
print("Examples of Misclassified Negative Reviews (Predicted as Positive):")
for i, original_index in enumerate(misclassified_negative_original_indices[:10]):
    relative_index_in_predicted = misclassified_negative_relative_indices[i]
    print(f"\nReview {i+1} (Original Index: {original_index}):")
    print("Original Text:", df.loc[original_index, 'Text'])
    print("Cleaned Text:", df.loc[original_index, 'Cleaned_Text'])
    print("True Sentiment:", df.loc[original_index, 'Sentiment'])
    print("Predicted Sentiment:", predicted_sentiment_for_negatives[relative_index_in_predicted])

# ---- Next Cell ----

def predict_sentiment(review):
    """
    Predicts the sentiment of a new review.

    Args:
        review (str): The raw text of the review.

    Returns:
        str: The predicted sentiment ('positive' or 'negative').
    """
    # Preprocess the review
    cleaned_review = review.lower()
    cleaned_review = cleaned_review.translate(str.maketrans('', '', string.punctuation))
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    processed_review = ' '.join(filtered_review)

    # Vectorize the processed review
    vectorized_review = vectorizer.transform([processed_review])

    # Predict sentiment
    prediction = model.predict(vectorized_review)

    return prediction[0]

# ---- Next Cell ----

# Example usage
new_review = "This product is absolutely fantastic! I love it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")

new_review_2 = "This is the worst product I have ever bought. It's a waste of money."
sentiment_2 = predict_sentiment(new_review_2)
print(f"The sentiment of the review is: {sentiment_2}")

# ---- Next Cell ----

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Start MLflow run for SVM
with mlflow.start_run(run_name="LinearSVC_Baseline"):
    # Initialize and train the LinearSVC model
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train_tfidf, y_train)

    # Evaluate the SVM model on the test set
    y_pred_svm_test = svm_model.predict(X_test_tfidf)

    accuracy_svm_test = accuracy_score(y_test, y_pred_svm_test)
    report_svm_test = classification_report(y_test, y_pred_svm_test, output_dict=True)
    
    print(f"SVM Model - Test Accuracy: {accuracy_svm_test:.4f}")
    print("SVM Model - Test Classification Report:\n", classification_report(y_test, y_pred_svm_test))

    # Log parameters
    mlflow.log_param("model_type", "LinearSVC")
    mlflow.log_param("C", 1.0)  # Default value
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_svm_test)
    mlflow.log_metric("precision_negative", report_svm_test['negative']['precision'])
    mlflow.log_metric("recall_negative", report_svm_test['negative']['recall'])
    mlflow.log_metric("f1_negative", report_svm_test['negative']['f1-score'])
    mlflow.log_metric("precision_positive", report_svm_test['positive']['precision'])
    mlflow.log_metric("recall_positive", report_svm_test['positive']['recall'])
    mlflow.log_metric("f1_positive", report_svm_test['positive']['f1-score'])
    
    # Log model
    signature = infer_signature(X_train_tfidf, svm_model.predict(X_train_tfidf))
    mlflow.sklearn.log_model(svm_model, "svm_model", signature=signature)

# Evaluate the SVM model on the validation set
y_pred_svm_val = svm_model.predict(X_val_tfidf)
accuracy_svm_val = accuracy_score(y_val, y_pred_svm_val)
print(f"SVM Model - Validation Accuracy: {accuracy_svm_val:.4f}")
report_svm_val = classification_report(y_val, y_pred_svm_val)
print("SVM Model - Validation Classification Report:\n", report_svm_val)

# ---- Next Cell ----

param_grid = {'C': [0.1, 1, 10]}

# ---- Next Cell ----

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Next Cell ----

from sklearn.model_selection import GridSearchCV

# Start MLflow run for Hyperparameter Tuning
with mlflow.start_run(run_name="LinearSVC_GridSearch"):
    # Instantiate a LinearSVC model
    svm_model = LinearSVC(random_state=42)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(svm_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_tfidf, y_train)

    # Print the best hyperparameters and the best cross-validation score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    # Get the best estimator from the grid search
    best_svm_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred_best_svm_test = best_svm_model.predict(X_test_tfidf)
    accuracy_best_svm_test = accuracy_score(y_test, y_pred_best_svm_test)
    report_best_svm_test = classification_report(y_test, y_pred_best_svm_test, output_dict=True)
    
    print(f"Best SVM Model - Test Accuracy: {accuracy_best_svm_test:.4f}")
    print("Best SVM Model - Test Classification Report:\n", classification_report(y_test, y_pred_best_svm_test))

    # Log parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "LinearSVC_tuned")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "accuracy")
    
    # Log metrics
    mlflow.log_metric("cv_best_score", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", accuracy_best_svm_test)
    mlflow.log_metric("precision_negative", report_best_svm_test['negative']['precision'])
    mlflow.log_metric("recall_negative", report_best_svm_test['negative']['recall'])
    mlflow.log_metric("f1_negative", report_best_svm_test['negative']['f1-score'])
    mlflow.log_metric("precision_positive", report_best_svm_test['positive']['precision'])
    mlflow.log_metric("recall_positive", report_best_svm_test['positive']['recall'])
    mlflow.log_metric("f1_positive", report_best_svm_test['positive']['f1-score'])
    
    # Log the best model
    signature = infer_signature(X_train_tfidf, best_svm_model.predict(X_train_tfidf))
    mlflow.sklearn.log_model(best_svm_model, "best_svm_model", signature=signature)
    
    # Log grid search results
    results_df = pd.DataFrame(grid_search.cv_results_)
    mlflow.log_table(results_df, "grid_search_results.json")

# Evaluate on validation set
y_pred_best_svm_val = best_svm_model.predict(X_val_tfidf)
accuracy_best_svm_val = accuracy_score(y_val, y_pred_best_svm_val)
print(f"Best SVM Model - Validation Accuracy: {accuracy_best_svm_val:.4f}")
report_best_svm_val = classification_report(y_val, y_pred_best_svm_val)
print("Best SVM Model - Validation Classification Report:\n", report_best_svm_val)

# ---- Next Cell ----

print("--- Model Performance Comparison ---")

print("\nInitial Multinomial Naive Bayes Model:")
print(f"Test Accuracy: {accuracy:.4f}")
print("Test Classification Report:\n", classification_report(y_test, y_pred))
print(f"Validation Accuracy: {accuracy_val:.4f}")
print("Validation Classification Report:\n", report_val)

print("\nTuned Linear SVM Model:")
print(f"Test Accuracy: {accuracy_best_svm_test:.4f}")
print("Test Classification Report:\n", classification_report(y_test, y_pred_best_svm_test))
print(f"Validation Accuracy: {accuracy_best_svm_val:.4f}")
print("Validation Classification Report:\n", report_best_svm_val)

# ---- Next Cell ----

# Choose a ProductId
product_id_to_analyze = df['ProductId'].iloc[100] # Using the ProductId of the first row as an example

# Filter the DataFrame for the chosen ProductId
product_reviews_df = df[df['ProductId'] == product_id_to_analyze].copy()

# Predict sentiment for each review in the filtered DataFrame
product_reviews_df['Predicted_Sentiment'] = product_reviews_df['Text'].apply(predict_sentiment)

# Calculate and print the distribution of predicted sentiments for the chosen product
print(f"Sentiment distribution for Product ID: {product_id_to_analyze}")
print(product_reviews_df['Predicted_Sentiment'].value_counts())

# ---- Next Cell ----

from nltk.probability import FreqDist

# Predict sentiment for the entire cleaned dataset using the best SVM model
predicted_sentiments = best_svm_model.predict(vectorizer.transform(X))

# Add the predicted sentiments as a new column to the df_sentiment DataFrame
df_sentiment['Predicted_Sentiment'] = predicted_sentiments

# Filter the DataFrame based on predicted sentiment
positive_reviews_df = df_sentiment[df_sentiment['Predicted_Sentiment'] == 'positive']
negative_reviews_df = df_sentiment[df_sentiment['Predicted_Sentiment'] == 'negative']

# Concatenate all cleaned text for each sentiment
all_positive_text = ' '.join(positive_reviews_df['Cleaned_Text'])
all_negative_text = ' '.join(negative_reviews_df['Cleaned_Text'])

# Find the frequency of words in positive and negative reviews
fdist_positive = FreqDist(word_tokenize(all_positive_text))
fdist_negative = FreqDist(word_tokenize(all_negative_text))

# Print the most common words
print("Most common words in positive reviews:")
print(fdist_positive.most_common(20))

print("\nMost common words in negative reviews:")
print(fdist_negative.most_common(20))

# ---- Next Cell ----

import matplotlib.pyplot as plt

# Create lists of words and frequencies for positive reviews
positive_words = [word for word, freq in fdist_positive.most_common(20)]
positive_freqs = [freq for word, freq in fdist_positive.most_common(20)]

# Create lists of words and frequencies for negative reviews
negative_words = [word for word, freq in fdist_negative.most_common(20)]
negative_freqs = [freq for word, freq in fdist_negative.most_common(20)]

# Create the bar plot for positive keywords
plt.figure(figsize=(12, 6))
plt.bar(positive_words, positive_freqs, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 20 Most Common Positive Keywords')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Create the bar plot for negative keywords
plt.figure(figsize=(12, 6))
plt.bar(negative_words, negative_freqs, color='salmon')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 20 Most Common Negative Keywords')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Log the final analysis results
with mlflow.start_run(run_name="Final_Analysis", nested=True):
    mlflow.log_metric("total_reviews", len(df_sentiment))
    mlflow.log_metric("positive_reviews", len(positive_reviews_df))
    mlflow.log_metric("negative_reviews", len(negative_reviews_df))
    mlflow.log_param("best_model", "LinearSVC_tuned")
    mlflow.log_metric("best_test_accuracy", accuracy_best_svm_test)