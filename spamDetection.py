import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

### Step 1: Load Dataset
dataframe = pd.read_csv("spam.csv")

print(dataframe.describe()) # to calculate the number of data for training and testing
#general rule of thumb: use 80% of data as training data set and remaining for testing


### Step 2: Split in to Training and Test Data
x = dataframe["EmailText"]
y = dataframe["Label"]

x_train, y_train = x[0:4457], y[0:4457]
x_test, y_test = x[4457:], y[4457:]

### Step 3: Extract Features
cv = CountVectorizer()
features = cv.fit_transform(x_train)


### Step 4: Build a model
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]} #param to tune for svm model

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features,y_train)

print(model.best_params_)

### Step 5: Test Accuracy

features_test = cv.transform(x_test)

print(model.score(features_test,y_test))
