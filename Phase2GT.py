import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data from CSV file
data = pd.read_csv('emails.csv')

# Assuming the CSV file has two columns: 'text' for email content and 'spam' for spam/ham label
emails = data['text'].values
labels = data['spam'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Predictions
y_pred = classifier.predict(X_test_counts)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Bar chart visualization
spam_count = sum(labels)
ham_count = len(labels) - spam_count

fig, ax = plt.subplots()
categories = ['Not Spam', 'Spam']
counts = [ham_count, spam_count]
ax.bar(categories, counts, color=['blue', 'red'])
ax.set_title('Number of Spam and Not Spam Emails')
ax.set_xlabel('Category')
ax.set_ylabel('Count')
plt.show()

# Print counts for verification
print(f"Number of Not Spam Emails: {ham_count}")
print(f"Number of Spam Emails: {spam_count}")
