import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# 1. Load the dataset
file_path = 'Iris.csv'
df = pd.read_csv(file_path)

# 2. Separate features (X) and target label (y)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# 3. Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Decision Tree Classifier
clf=DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 5. Make predictions on test set
y_pred = clf.predict(X_test)

# 6. Calculate & display performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("=== MODEL EVALUATION METRICS ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}\n")

print("=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))
