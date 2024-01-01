import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


#The "nbModel" function trains a Gaussian NB model on a dataset uploaded by the user, then returns the evaluation results of the NB model's performance

def nbModel(file_path):
    # Load the iris dataset from the provided file
    iris_data = pd.read_csv(file_path) 

    # Prepare the Data for training
    X = iris_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
    y = iris_data['variety']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

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

    # Save the trained model to a file
    joblib.dump(nb_model, 'naive_bayes_model.joblib')

    # Return the results as a dictionary
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

    return results



#----------------------------------------------------------------------------------------------------------------------------




#The "nbIrisClassifier" function uses the trained Gaussian Nb model to classify an instance of iris data entered by the user

def nbIrisClassifier(sepal_length, sepal_width, petal_length, petal_width):
    try:
        # Load the trained Naive Bayes model
        nb_model = joblib.load('naive_bayes_model.joblib')
    except FileNotFoundError:
        print("Model file not found. Make sure to train the model first.")
        return None


    # Prepare the input data for prediction with feature names
    new_instance = pd.DataFrame({
        'sepal.length': [sepal_length],
        'sepal.width': [sepal_width],
        'petal.length': [petal_length],
        'petal.width': [petal_width]
    }, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])


    # Use the trained Naive Bayes model to make predictions
    prediction = nb_model.predict(new_instance)

    # Return the result of the classification
    return prediction[0]

