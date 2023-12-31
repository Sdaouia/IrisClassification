import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris_data = pd.read_csv('iris_dataset.csv') 

# Prepare the Data for training
X = iris_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris_data['variety']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#train_test_split : shuffles the data before splitting, 
#random_state parameter : ensures reproducibility in the shuffling process. 