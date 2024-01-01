from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
import numpy as np

# "SVM" function take as argument the dataset file_path uploaded by the user,
# then apply SVM model on the datase, 
# finally return the evaluation results. 
def SVM(file_path):

    
# Import dataset from the file_path
    dataset = pd.read_csv(file_path)
    
# Declare feature Metrix(X) and target vector(Y)
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values
    
    
# Split Data into separate trainset and testset 
# with 20% for testset 
    X_train ,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)
    

# Generating The SVM Model with default hyperparameters
# kernal = rbf , C = 1.0, gamma = auto
    svm_model = SVC()
   

#declare parameters for hyperparameter tuning
    param_grid =  [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
                    {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,'auto', 'scale']},
                    {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05]} 
                  ]
    
# Perform grid_search with cross_validation(cv) = 5.
# GridSearchCV helps to identify the parameters 
# that will improve the performance for this particular model.
   
    grid_search = GridSearchCV(svm_model, param_grid, cv = 5)
  
    
# Fit classifier to training set
    grid_search.fit(X_train, Y_train)


# Declar the estimator that was chosen by the GridSearch thats give the best
# the output is: SVC(parameters)
    best_svm_model = grid_search.best_estimator_
    

# Make predictions on test_set
    y_pred = best_svm_model.predict(X_test)


# Evaluate the Model and Return the results as a dictionary
    results = {
        'accuracy': best_svm_model.score(X_test, Y_test),
        'classification_report': classification_report(Y_test, y_pred),
        'confusion_matrix': confusion_matrix(Y_test, y_pred)
        }
    
# save the trained model to a file 
    joblib.dump(best_svm_model, 'trained_SVM_model.joblib')
    
    
    return results
 


#//////////////////////////////////////////////////////////////////////////



# "SVM_classification" function take as arguments features values of the new instance entered by the user,
#  then uses the trained SVM model to classify the new instance of irisdata.
def SVM_classification(sepal_length, sepal_width, petal_length, petal_width):
 
    
    SVM_model = joblib.load('trained_SVM_model.joblib')

# Prepare the new instance input in a colunm array
    new_instance = np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1, -1)
    
# Classified the new instance using SVM model
    prediction = SVM_model.predict(new_instance) 
    
# Return the predict class name  
    return prediction[0]
    