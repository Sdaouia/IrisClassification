from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def SVM_classification(file_path, sepal_length, sepal_width, petal_length, petal_width):
    
    df = pd.read_csv(file_path)
    
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values
    
    
# Splitting Data into 20% for training
    X_train ,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    
    
# Generating The SVM Modele with default hyperparameters
    svm_model = SVC()
    
# fit classifier to training set
    svm_model.fit(X_train, Y_train)
    

# classified a new instance   
    new_instance = np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1, -1)
    prediction = svm_model.predict(new_instance) 
   
    return prediction