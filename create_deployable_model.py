"""
Create a deployable model that accepts raw text input by rebuilding the
vectorizer on the original training split and training LinearSVC with the
best hyperparameters from the MLflow GridSearch run. The saved model
handles preprocessing + vectorization internally and exposes a clean
pyfunc interface: input DataFrame must contain a 'text' column.
"""

import os
import string
import warnings
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

warnings.filterwarnings("ignore")

# Ensure NLTK assets are present
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set tracking URI to local mlruns
TRACKING_URI = r"file:///C:/Users/Atharv/Desktop/projectss/mlops/mlops_project_group1/src/mlruns"
mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()
experiment = client.get_experiment_by_name("Sentiment_Analysis")
if experiment is None:
    print("❌ Experiment 'Sentiment_Analysis' not found")
    raise SystemExit(1)

# Fetch best GridSearch run to read best params
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string='tags.`mlflow.runName` = "LinearSVC_GridSearch"',
    order_by=["metrics.test_accuracy DESC"],
    max_results=1,
)
if not runs:
    print("❌ No LinearSVC_GridSearch run found!")
    raise SystemExit(1)

best_run = runs[0]
best_C = float(best_run.data.params.get("C", 1.0))
print(f"✅ Best GridSearch run: {best_run.info.run_id} | best C = {best_C}")

# Load and preprocess dataset (exactly as in train.py)
data_path = r"C:\Users\Atharv\Desktop\projectss\mlops\mlops_project_group1\src\Reviews.csv"
df = pd.read_csv(data_path)

df['Text'] = df['Text'].fillna('')
df['Cleaned_Text'] = df['Text'].str.lower()
df['Cleaned_Text'] = df['Cleaned_Text'].str.translate(str.maketrans('', '', string.punctuation))

stop_words = set(stopwords.words('english'))
df['Cleaned_Text'] = df['Cleaned_Text'].apply(word_tokenize)
df['Cleaned_Text'] = df['Cleaned_Text'].apply(lambda x: [word for word in x if word not in stop_words])
df['Cleaned_Text'] = df['Cleaned_Text'].apply(lambda x: ' '.join(x))

df['Sentiment'] = df['Score'].apply(
    lambda score: 'positive' if score in [4, 5] else ('negative' if score in [1, 2] else None)
)
df_sentiment = df.dropna(subset=['Sentiment'])

X = df_sentiment['Cleaned_Text']
y = df_sentiment['Sentiment']

# Recreate same splits as training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Fit vectorizer on X_train ONLY (consistent with training)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train LinearSVC with best C
svm_model = LinearSVC(random_state=42, C=best_C)
svm_model.fit(X_train_tfidf, y_train)

# Evaluate for sanity
test_acc = accuracy_score(y_test, svm_model.predict(X_test_tfidf))
print(f"✅ Rebuilt model Test Accuracy: {test_acc:.4f}")


class SentimentModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        cleaned = text.lower()
        cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(cleaned)
        filtered = [w for w in tokens if w not in self.stop_words]
        return ' '.join(filtered)

    def predict(self, context, model_input):
        # Expect a DataFrame with a 'text' column
        if isinstance(model_input, pd.DataFrame):
            texts = model_input['text'].astype(str).values
        else:
            # Fallback: treat input as list-like of strings
            texts = list(model_input)

        cleaned_texts = [self.preprocess_text(t) for t in texts]
        X_vec = self.vectorizer.transform(cleaned_texts)
        preds = self.model.predict(X_vec)
        return preds


print("\n💾 Logging deployable model to MLflow…")
input_schema = Schema([ColSpec("string", "text")])
output_schema = Schema([ColSpec("string")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

wrapped_model = SentimentModelWrapper(vectorizer, svm_model)

with mlflow.start_run(run_name="Deployable_SVM_Model_Rebuilt") as run:
    mlflow.log_metric("rebuilt_test_accuracy", test_acc)
    mlflow.log_param("model_type", "LinearSVC_with_preprocessing")
    mlflow.log_param("C", best_C)

    mlflow.pyfunc.log_model(
        artifact_path="sentiment_model",
        python_model=wrapped_model,
        signature=signature,
        input_example=pd.DataFrame({"text": ["This product is amazing!"]}),
    )

    new_run_id = run.info.run_id

print("\n✅ Deployable model saved!")
print(f"   Run ID: {new_run_id}")
print("\n🚀 Serve with:")
print(f"   mlflow models serve -m runs:/{new_run_id}/sentiment_model -p 5001 --env-manager=local")

