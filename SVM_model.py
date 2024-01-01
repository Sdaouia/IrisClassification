from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

def SVM(file_path):
    df = pd.read_csv(file_path)
    
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values
    
    
# Splitting Data into 20% for training
    X_train ,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    
    
# Generating The SVM Modele with default hyperparameters
    svm_model =SVC()
    
# fit classifier to training set
    svm_model.fit(X_train, Y_train)
    
# make predictions on test set
    y_pred = svm_model.predict(X_test)
    
# Evaluating the Model
    accuracy = accuracy_score(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
    conf_matrix = confusion_matrix(Y_test, y_pred)    
    
    return accuracy, report, conf_matrix 
    
    