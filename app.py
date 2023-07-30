"""## 🎁 Packages:"""
import nltk
import spacy
import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import torch
import re

# Data Loading
from sklearn.utils import shuffle

# Setting the file paths
file1 = "fake reviews dataset.csv"
file2 = "The Capstone Fake Reviews Dataset GPT4.csv"

# Loading first csv file
df1 = pd.read_csv(file1)

# Filtering rows where 'Label' equals 'OR' or 'CG'
df1_OR = df1[df1['Label'] == 'OR']
df1_CG = df1[df1['Label'] == 'CG']

# Randomly selecting 5000 rows where 'Label' equals 'OR'
df1_OR_sample = df1_OR.sample(n=5000, random_state=1)

# Randomly selecting 2500 rows where 'Label' equals 'CG'
df1_CG_sample = df1_CG.sample(n=2500, random_state=1)

# Concatenating the two subsets
df1_sample = pd.concat([df1_OR_sample, df1_CG_sample])

# Loading second csv file
df2 = pd.read_csv(file2)

# Concatenating the two dataframes
df = pd.concat([df1_sample, df2])

# Shuffling the combined dataframe
df = shuffle(df, random_state=1)

# Resetting index of the combined dataframe
df.reset_index(drop=True, inplace=True)

# Keeping only the 'Category', 'Text', and 'Label' columns
df = df[['Category', 'Text', 'Label']]


"""## 🔪 Splitting the dataset"""

from sklearn.model_selection import train_test_split

# Defining the X and y based on the joined text
X = df['Text']
y = df['Label']

# Train test split time!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from IPython.display import display, HTML

display(HTML(f"<h1 style='font-size:70px; color: #106EC7;'>Training Data Only:</h1>"))

"""## 🧹 Data cleaning/checking:

### 0️⃣ Looking for nulls:
"""

# Converting training data to DataFrame
train_df = pd.DataFrame({'Text': X_train, 'Label': y_train})

# Handling missing values
missing_counts = train_df.isnull().sum()
print(missing_counts)  # to visualize missing values
train_df.dropna(inplace=True)  # or use other appropriate handling method

# Separating features and target after cleaning
X_train = train_df['Text']
y_train = train_df['Label']

"""#### 🔨 Deleting nulls:"""

# Dropping rows where 'Label' is missing from train_df
train_df.dropna(subset=['Label'], inplace=True)

# Checking the count of missing values
missing_counts = train_df.isnull().sum()
print(missing_counts)  # to visualize missing values

# Separate features and target after cleaning
X_train = train_df['Text']
y_train = train_df['Label']

"""### 🧑‍🤝‍🧑 Looking for duplicates"""

# Looking for duplicates in the training data
duplicates_train = train_df.duplicated()

# Counting the number of duplicates in the training data
duplicate_count_train = duplicates_train.sum()

print(f'There are {duplicate_count_train} duplicate rows in the training data.')

"""#### 🔨 Deleting duplicates:"""

# Dropping duplicate rows in the training data
train_df.drop_duplicates(inplace=True)

# Checking again the count of duplicates in the training data to verify
duplicate_count_train = train_df.duplicated().sum()
print(f'There are {duplicate_count_train} duplicate rows in the training data after removal.')

# Separate features and target after cleaning
X_train = train_df['Text']
y_train = train_df['Label']

"""### 🔡 Changing to lowercase and expanding contractions:"""

pip install contractions

import contractions

# Creating a new DataFrame with the training data
# Converting the text to lowercase and expand contractions
train_df_processed = train_df.copy()
train_df_processed['Text'] = train_df['Text'].str.lower().apply(lambda x: contractions.fix(x))

# Separating features and target after processing again...
X_train = train_df_processed['Text']
y_train = train_df_processed['Label']

"""### 🐝 Detecting spelling errors:

I didn't do spell corrections. Due to the nature of the dataset, especially the informal nature of the Amazon product reviews, spell correction might cause more errors more than it's worth.

### 💻 Looking for URLs in reviews:
"""

# Looking for URLs in the training data
mask = train_df_processed['Text'].str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', regex=True)

# Counting the number of True values
num_urls = mask.sum()

# Printing the number of rows with URLs in the training data
print(f"Number of rows with URLs in the training data: {num_urls}")

"""The number of URLs in the 'Text' column was relatively insignificant here. Moreover, they might potentially be a good indicator of fake reviews. So, I left them in.

### 📙 Looking for non-English characters:
"""

# Looking for non-English characters in the training data
mask = train_df_processed['Text'].str.contains(r'[^\x00-\x7F]', regex=True)

# Counting the number of True values in the mask
num_non_english_chars = mask.sum()

# Printing the number of rows with non-English characters in the training data
print(f"Number of rows with non-English characters in the training data: {num_non_english_chars}")

"""The number of non-English characters is relatively small for the dataset. While deleting might be an easy option, perhaps due to caution, they likely would be an indicator towards an authentic (real ) review. So, I've chosen to leave them in the data.

### 🖱️Looking for HTML tags
"""

# Looking for HTML tags in the training data
mask = train_df_processed['Text'].str.contains(r'<.*?>', regex=True)

# Counting the number of True values in the mask
num_html_tags = mask.sum()

# Printing the number of rows with HTML tags in the training data
print(f"Number of rows with HTML tags in the training data: {num_html_tags}")

"""### I tried deleting these, but impacted the model. So, they are contributing to the model. Thus, I've let them in.

#### 🔨 Removing HTML tags, but leaving the content within those tags:
"""

import re

# Removing HTML tags in the training data
train_df_processed['Text'] = train_df_processed['Text'].apply(lambda x: re.sub('<.*?>', '', x))

# Updating X_train after processing again...
X_train = train_df_processed['Text']

"""### 👩‍💻 Encoding the 'Label' column:

CG 'computer generated' = 1 and OR 'original (real) review' = 0
"""

# Mapping 'Label' values to binary in the training data
train_df_processed['Label'] = train_df_processed['Label'].map({'CG': 1, 'OR': 0})

# Updating y_train after processing again...
y_train = train_df_processed['Label']

"""#### ☑️ Checking the DataFrame:"""

train_df_processed.sample(5)

"""# 🏭 Pre-processing

## 🪙 Tokenisation: Pre-processing the 'Text' column
"""

import nltk
nltk.download('punkt')  # Downloading the Punkt tokenizer

# Tokenizing the text and add it as a new column in the training data
train_df_processed['Tokenized_Text'] = train_df_processed['Text'].apply(nltk.word_tokenize)

# Updating X_train after processing, yet again...
X_train = train_df_processed[['Text', 'Tokenized_Text']]

"""## 🛑 Stop Words

I did not remove stop words. I initially did; however, removing stop words negatively impacted my model.

## 🍋 Lemmatisation - noise reduction
"""

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # Downloading the WordNet lemmatizer

# Initialising the lemmatizer
lemmatizer = WordNetLemmatizer()

# Applying the lemmatizer to the training data
train_df_processed['Lemmatized_Text'] = train_df_processed['Tokenized_Text'].apply(lambda row: [lemmatizer.lemmatize(word) for word in row])

# Updating X_train after processing, as usual...
X_train = train_df_processed[['Text', 'Tokenized_Text', 'Lemmatized_Text']]

"""## 🔗 Joining the tokens back into a string before applying the TfidfVectorizer:"""

# Joining the lemmatized tokens back into a single string in the training data
train_df_processed['Lemmatized_Text_Joined'] = train_df_processed['Lemmatized_Text'].apply(' '.join)

# Update X_train after processing, the routine...
X_train = train_df_processed[['Text', 'Tokenized_Text', 'Lemmatized_Text', 'Lemmatized_Text_Joined']]

"""## 📐 Vectorisation"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Implementing the TF-IDF transformer
vectorizer = TfidfVectorizer()

# Fitting and transforming on the training data
X_train_tfidf = vectorizer.fit_transform(train_df_processed['Lemmatized_Text_Joined'])

# Converting the results back to dataframes, including the label column
df_train_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
df_train_tfidf['Label'] = y_train.values

"""## 📚 Stacking ML Model"""

pip install -q lightgbm

from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Define the base models
base_models = [
    ('ridge_clf', RidgeClassifier()),
    ('svc', SVC(kernel='linear', probability=True)),
    ('extra_trees_clf', ExtraTreesClassifier(n_estimators=100, random_state=42))
]

# Meta model
meta_model = LogisticRegression()

# Build the stacking classifier
stacking_clf= StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the stacking classifier
stacking_clf.fit(X_train_tfidf, y_train)

"""## 🎯 Y-train Predictions on test data:"""

from sklearn.metrics import accuracy_score

# Predict the training data
y_train_pred = stacking_clf.predict(X_train_tfidf)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f'Training accuracy: {train_accuracy:.2f}')

from IPython.core.display import display, HTML

# Assuming 'accuracy' variable holds the accuracy score
accuracy = 100  # replace this with the actual variable holding the accuracy score

display(HTML(f"<h1 style='font-size:40px; color: #FF6EC7;'>Accuracy: {accuracy}% (Um, well, on the training data...)</h1>"))

display(HTML(f"<h1 style='font-size:70px; color: #106EC7;'>Test Data Only:</h1>"))

"""Below is the test data. I've divided up the training and test data completely to avoid any possibility of leakage. The concept is to simulate what 'raw' data would be like to enter and be assessed by the model.

# 🧹Cleaning:
"""

# Converting testing data to DataFrame for easier data manipulation
test_df = pd.DataFrame({'Text': X_test, 'Label': y_test})

#DELETING NULLS:
# Dropping rows where 'Label' is missing from test_df
test_df.dropna(subset=['Label'], inplace=True)

# Checking the count of missing values
missing_counts = test_df.isnull().sum()

# Separate features and target after cleaning
X_test = test_df['Text']
y_test = test_df['Label']

# -----------------------------------------------------------
#DELETING DUPLICATES:
# Dropping duplicate rows in the testing data
test_df.drop_duplicates(inplace=True)

# Checking again the count of duplicates in the testing data to verify
duplicate_count_test = test_df.duplicated().sum()

# Separate features and target after cleaning
X_test = test_df['Text']
y_test = test_df['Label']

#-------------------------------------------------------------
#CONTRACTIONS AND LOWERCASE:
# Creating a new DataFrame with the testing data
# Converting the text to lowercase and expand contractions
test_df_processed = test_df.copy()
test_df_processed['Text'] = test_df['Text'].str.lower().apply(lambda x: contractions.fix(x))

# Separating features and target after processing again...
X_test = test_df_processed['Text']
y_test = test_df_processed['Label']

#-------------------------------------------------------------
#HTML TAG REMOVAL:
# Removing HTML tags in the testing data
test_df_processed['Text'] = test_df_processed['Text'].apply(lambda x: re.sub('<.*?>', '', x))

# Updating X_test after processing again...
X_test = test_df_processed['Text']

#--------------------------------------------------------------
#ENCODING THE TARGET FOR TEST DATA

# Mapping 'Label' values to binary in the test data
test_df['Label'] = test_df['Label'].map({'CG': 1, 'OR': 0})

# Updating y_test after processing...
y_test = test_df['Label']

"""# 🏭 Pre-processing"""

#TOKENISATION
# Tokenizing the text and add it as a new column in the testing data
test_df_processed['Tokenized_Text'] = test_df_processed['Text'].apply(nltk.word_tokenize)

# Updating X_test after processing, yet again...
X_test = test_df_processed[['Text', 'Tokenized_Text']]

#--------------------------------------------------------------------------------
#LEMMATISATION
# Applying the lemmatizer to the testing data
test_df_processed['Lemmatized_Text'] = test_df_processed['Tokenized_Text'].apply(lambda row: [lemmatizer.lemmatize(word) for word in row])

# Updating X_test after processing, as usual...
X_test = test_df_processed[['Text', 'Tokenized_Text', 'Lemmatized_Text']]

#-------------------------------------------------------------------------------
#JOINING
# Joining the lemmatized tokens back into a single string in the testing data
test_df_processed['Lemmatized_Text_Joined'] = test_df_processed['Lemmatized_Text'].apply(' '.join)

# Update X_test after processing, the routine...
X_test = test_df_processed[['Text', 'Tokenized_Text', 'Lemmatized_Text', 'Lemmatized_Text_Joined']]

#-------------------------------------------------------------------------------
#VECTORISATION
# Transforming the testing data using the TF-IDF transformer
X_test_tfidf = vectorizer.transform(test_df_processed['Lemmatized_Text_Joined'])

# Converting the results back to dataframes, including the label column
df_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
df_test_tfidf['Label'] = y_test.values

"""# ✈️ The Model for Test Data:"""

#THE MODEL - THE TRAINED STACKING CLASSIFIER
# The model's already been trained on the training data, so I didn't include here.

# Predict the test data
y_test_pred = stacking_clf.predict(X_test_tfidf)

# Calculate the test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Test accuracy: {test_accuracy:.2f}')

#--------------------------------------------------------------------------
#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Computing the confusion matrix for test data
cm = confusion_matrix(y_test, y_test_pred)

# Creating a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(cm, index=['Actual Real', 'Actual Fake'], columns=['Predicted Real', 'Predicted Fake'])

plt.figure(figsize=(7,5))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Real vs. Fake Product Reviews -- A miracle!')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print("\n")
#---------------------------------------------------------------------------
# ROC CURVE
from sklearn.metrics import roc_curve, auc
# Predict probabilities for test data using the already trained stacking classifier
y_proba = stacking_clf.predict_proba(X_test_tfidf)

# We need scores for positive class to plot ROC curve
y_scores = y_proba[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test Data)')
plt.legend(loc="lower right")
plt.show()

"""# 🤖 Anvil app (for raw input):

### Creating the pipeline for Anvil & creating the joblib dump:
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import re
import pandas as pd
import nltk
import joblib

nltk.download('punkt')
nltk.download('wordnet')

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convert to DataFrame
        df = pd.DataFrame({'Text': X})

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Expand contractions and convert to lowercase
        df['Text'] = df['Text'].str.lower().apply(lambda x: contractions.fix(x))

        # Remove HTML tags
        df['Text'] = df['Text'].apply(lambda x: re.sub('<.*?>', '', x))

        # Tokenize text
        df['Tokenized_Text'] = df['Text'].apply(word_tokenize)

        # Lemmatize tokens
        df['Lemmatized_Text'] = df['Tokenized_Text'].apply(lambda row: [self.lemmatizer.lemmatize(word) for word in row])

        # Join lemmatized tokens
        df['Lemmatized_Text_Joined'] = df['Lemmatized_Text'].apply(' '.join)

        return df['Lemmatized_Text_Joined']

# Create a pipeline
capstone_pipe = Pipeline([
    ('preprocessor', Preprocessor()),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', stacking_clf)  # replace this with your trained classifier
])
# Streamlit here

import streamlit as st

st.title("Stacking Classifier Model for Text Data")

# Capture the text input
user_input = st.text_input("Enter text here")

# If the button is pressed, execute the model training and prediction
if st.button("Predict"):

    # Predict the entered text
    prediction = capstone_pipe.predict([user_input])

    # Display the prediction
    st.write(f"The prediction for the entered text is: {prediction}")

