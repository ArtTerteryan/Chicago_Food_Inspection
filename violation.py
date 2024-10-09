import pandas as pd
import numpy as np
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


df = pd.read_csv("/home/artur/Desktop/Food_Inspections.csv")

df = df[['Violations', 'Results']].dropna(subset=['Results'])
df['Violations'].fillna("NO VIOLATION", inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text) 
    return text

df['Violations'] = df['Violations'].apply(preprocess_text)

results_mapping = {result: idx for idx, result in enumerate(df['Results'].unique())}
df['Results'] = df['Results'].map(results_mapping)

tfidf = TfidfVectorizer(max_features=500)  
X_tfidf = tfidf.fit_transform(df['Violations'])

top_words = ['word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10']

def create_feature_for_word(df, word):
    feature_name = f"contains_{word}"
    df[feature_name] = df['Violations'].apply(lambda x: 1 if word in x else 0)

for word in top_words:
    create_feature_for_word(df, word)

X_combined = pd.DataFrame(X_tfidf.toarray(), index=df.index)
for word in top_words:
    feature_name = f"contains_{word}"
    X_combined[feature_name] = df[feature_name]

X_combined.columns = X_combined.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X_combined, df['Results'], test_size=0.3, random_state=42)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_pred = tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


feature_importances = pd.Series(tree_model.feature_importances_, index=X_combined.columns)
print(feature_importances.nlargest(10))


feature_importances_sorted = feature_importances.sort_values(ascending=True).tail(10)

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index, palette="viridis")
plt.title('Top 10 Feature Importances in Decision Tree Model', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()