import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

# Load the iris dataset
iris_data = pd.read_csv('iris_dataset.csv') 

# Prepare the Data for training
X = iris_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris_data['variety']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#train_test_split : shuffles the data before splitting, 
#random_state parameter : ensures reproducibility in the shuffling process. 

# Initialize Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)



#Evaluate the performance of the model

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy of the Naive Baye model is {accuracy:.2f}\nThis means that it can correctly predict the type of the Iris flower {round(accuracy * 100)}% of the time.\n")

#calculate classification report
print('Classification Report:\n', classification_report(y_test, y_pred))


# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(conf_matrix, columns=nb_model.classes_, index=nb_model.classes_))
