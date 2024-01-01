import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def naiveBayesModel(file_path):
    # Load the iris dataset from the provided file
    iris_data = pd.read_csv(file_path) 

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



    # Evaluate the performance of the model

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate classification report
    report = classification_report(y_test, y_pred)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Return the results as a dictionary
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

    return results
