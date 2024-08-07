
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from warnings import filterwarnings
import pickle

filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('data.csv', sep=',', on_bad_lines='skip', engine='python')

# Fill NaN values in passwords with empty strings and convert to string type
data["password"] = data["password"].fillna('').astype(str)

# Define a function to find punctuation in passwords
def find_semantics(row):
    for char in row:
        if char in string.punctuation:
            return 1
    return 0

# Apply the function to create a new column for passwords containing special characters
data["special_char"] = data["password"].apply(find_semantics)

# Create new feature columns based on password characteristics
data["length"] = data["password"].str.len()

def freq_lowercase(row):
    return len([char for char in row if char.islower()]) / len(row) if len(row) > 0 else 0

def freq_uppercase(row):
    return len([char for char in row if char.isupper()]) / len(row) if len(row) > 0 else 0

def freq_numerical_case(row):
    return len([char for char in row if char.isdigit()]) / len(row) if len(row) > 0 else 0

def freq_special_case(row):
    special_chars = [char for char in row if not char.isalpha() and not char.isdigit()]
    return len(special_chars) / len(row) if len(row) > 0 else 0

data["lowercase_freq"] = np.round(data["password"].apply(freq_lowercase), 3)
data["uppercase_freq"] = np.round(data["password"].apply(freq_uppercase), 3)
data["digit_freq"] = np.round(data["password"].apply(freq_numerical_case), 3)
data["special_char_freq"] = np.round(data["password"].apply(freq_special_case), 3)

# Shuffle the dataframe
dataframe = data.sample(frac=1)

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer="char")
X = vectorizer.fit_transform(list(dataframe["password"]))

df2 = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df2["length"] = dataframe['length']
df2["lowercase_freq"] = dataframe['lowercase_freq']
df2["uppercase_freq"] = dataframe['uppercase_freq']
df2["digit_freq"] = dataframe['digit_freq']
df2["special_char_freq"] = dataframe['special_char_freq']

# Prepare target variable
y = dataframe["strength"]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.20)

# Logistic Regression model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(multi_class="multinomial")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100}%")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
with open('password_strength_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
    
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

