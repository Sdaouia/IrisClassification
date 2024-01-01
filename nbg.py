from tkinter import *
from tkinter import filedialog
import csv 
from tkinter import messagebox
from PIL import ImageTk,Image



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

### DONE WITH BAYES + GUI

######################################################################  daouia 's functions
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
    joblib.dump(nb_model, 'trained_NB_model.joblib')

    # Return the results as a dictionary
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

    return results


def nbIrisClassifier(sepal_length, sepal_width, petal_length, petal_width):
    
    # Load the trained Naive Bayes model
    nb_model = joblib.load('trained_NB_model.joblib')
    

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





###################################################################### KATIA 'S GUI
global opened_file
opened_file=None

def openFiles():
    root.filename=filedialog.askopenfilename(initialdir="/",title="select file",filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    global opened_file
    opened_file=root.filename
    if root.filename:
        #print(root.filename)
        upload_button.config(text="Iris.csv",bg='#DCDCDC')
        # Read the content of the CSV file
        

def training(model_value,opened_file):

    print("Training with",model_value)

    #with open(opened_file, 'r') as file:
            #reader = csv.reader(file)
            #for row in reader:
                #print(row)
    
    #call the model training
    if(model_value==2):
        print("before calling nb_model")
        print("path=",opened_file)
        result=nbModel(opened_file)
        #print(result['accuracy'])  
        #print(result['classification_report']) 
        #print(result['confusion_matrix']) 

    #go to page 2
    move_to_second_page(result,model_value)          
    

def check_uploaded_file(model_value):
      if(opened_file is not None):
                print("you entred your dataset successfully")
                training(model_value,opened_file)
      else:
          messagebox.showerror("Error", "Please enter your data set ! ")

def on_submit(sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button,model_value):
    entry_values = [entry.get() for entry in [sepal_length_input,sepal_width_input,petal_length_input,petal_width_input]]
    try:
        float_values = [float(value) for value in entry_values]
        print("Float values:", float_values,"with model",model_value)
        forgetForme(sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button)
        #call classification function(,,,,model_value) four value
        if(model_value==2):
          print("before calling  nb classifier")
          variety=nbIrisClassifier(float(sepal_length_input.get()), float(sepal_width_input.get()), float(petal_length_input.get()), float(petal_width_input.get()))
          print(variety)
          move_to_forth_page(model_value,variety)
          
    except ValueError:
        messagebox.showerror("Error", "Please enter valid float numbers.")


def move_to_second_page(result,model_value):
      upload_label.grid_forget()
      upload_button.grid_forget()
      choose_model.grid_forget()
      svm_option.grid_forget()
      bayes_option.grid_forget()
      decision_tree_option.grid_forget()
      train_button.grid_forget()

      accuracy=Label(root, text=f"The accuracy of the Naive Bayes model is {result['accuracy']:.2f}\nThis means that it can correctly predict the type of the Iris flower {round(result['accuracy'] * 100)}% of the time.\n\n",width=0 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')
      classification_report_label=Label(root, text="Classification report :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')
      classification_report=Label(root, text=result['classification_report'],width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')
      confusion_matrix_label=Label(root, text="\n\nConfusion matrix :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')
      confusion_matrix=Label(root, text=pd.DataFrame(result['confusion_matrix'], columns=['Setosa', 'Versicolor', 'Virginica'], index=['Setosa', 'Versicolor', 'Virginica']),width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')
      instance_button=Button(root, text="Entre an instance",width=30 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:move_to_third_page(accuracy,classification_report_label,classification_report,confusion_matrix_label,confusion_matrix,instance_button,model_value))


      accuracy.grid(row=0, column=0, padx=10, pady=10, sticky="e")
      classification_report_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
      classification_report.grid(row=2, column=0, padx=10, pady=10, sticky="e")
      confusion_matrix_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
      confusion_matrix.grid(row=3, column=0, padx=10, pady=10, sticky="e")
      instance_button.grid(row=4, column=0, padx=10, pady=10)


def move_to_third_page(accuracy,classification_report_label,classification_report,confusion_matrix_label,confusion_matrix,instance_button,model_value):
    accuracy.grid_forget()
    classification_report_label.grid_forget()
    confusion_matrix_label.grid_forget()
    classification_report.grid_forget()
    confusion_matrix.grid_forget()
    instance_button.grid_forget()

    #Labels
    sepal_length_label = Label(root, text="Sepal Length",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    sepal_length_label = Label(root, text="Sepal Length",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    sepal_width_label = Label(root, text="Sepal Width",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    petal_length_label = Label(root, text="Petal Length",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    petal_width_label = Label(root, text="Petal Width",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')

    #Inputs
    sepal_length_input = Entry(root, width=20 , font=('Helve', 16))
    #sepal_length_input.insert(0,"Entre sepal length :")

    sepal_width_input = Entry(root, width=20 , font=('Helvetica', 16))
    petal_length_input = Entry(root, width=20 , font=('Helvetica', 16))
    petal_width_input = Entry(root, width=20 , font=('Helvetica', 16))

    #Buttons
    classity_button = Button(root, text="Classify Now!",width=30 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:on_submit(sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button,model_value))
    

  
      # Grid Positioning
    sepal_length_label.grid(row=0, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_length_input.grid(row=0, column=0, padx=10, pady=(10,7))
 
    sepal_width_label.grid(row=1, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_width_input.grid(row=1, column=0, padx=10, pady=(10,7))

    petal_length_label.grid(row=2, column=0, padx=10, pady=(10,7), sticky="e")
    petal_length_input.grid(row=2, column=0, padx=10, pady=(10,7))

    petal_width_label.grid(row=3, column=0, padx=10, pady=(10,7), sticky="e")
    petal_width_input.grid(row=3, column=0, padx=10, pady=(10,7))

    classity_button.grid(row=4, column=0, padx=(10,10), pady=10)

def forgetForme(sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button):

    sepal_length_input.grid_forget()
    sepal_length_label.grid_forget()
    sepal_width_label.grid_forget()
    sepal_width_input.grid_forget()
    petal_length_input.grid_forget()
    petal_length_label.grid_forget()
    petal_width_label.grid_forget()
    petal_width_input.grid_forget()
    classity_button.grid_forget()

def move_to_forth_page(model_value,variety):
     variety_label = Label(root, text=f"the instance is {variety}",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
     image_label =Label(root)
     try_again_button=Button(root, text="Try Again",width=10 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:come_back_to_third(variety_label,image_label,exit_button,try_again_button,model_value))
     exit_button=Button(root, text="Exit",width=10 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=root.destroy)
     
     
     img = Image.open("setosa.jpg")
     resized_img = img.resize((300, 300))
     photo = ImageTk.PhotoImage(resized_img)
     image_label.config(image=photo)
     image_label.image = photo 
    
    
     variety_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
     image_label.grid(row=1, column=0, padx=10, pady=10)
     try_again_button.grid(row=2, column=0, padx=(5,0), pady=10)
     exit_button.grid(row=2, column=1, padx=(0,5), pady=10)

def come_back_to_third(variety_label,image_label,exit_button,try_again_button,model_value):
    
    variety_label.grid_forget()
    image_label.grid_forget()
    exit_button.grid_forget()
    try_again_button.grid_forget()

    #Labels
    sepal_length_label = Label(root, text="Sepal Length",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    sepal_length_label = Label(root, text="Sepal Length",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    sepal_width_label = Label(root, text="Sepal Width",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    petal_length_label = Label(root, text="Petal Length",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    petal_width_label = Label(root, text="Petal Width",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')

    #Inputs
    sepal_length_input = Entry(root, width=20 , font=('Helve', 16))
    #sepal_length_input.insert(0,"Entre sepal length :")

    sepal_width_input = Entry(root, width=20 , font=('Helvetica', 16))
    petal_length_input = Entry(root, width=20 , font=('Helvetica', 16))
    petal_width_input = Entry(root, width=20 , font=('Helvetica', 16))

    #Buttons
    classity_button = Button(root, text="Classify Now!",width=30 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:on_submit(sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button,model_value))
    

  
      # Grid Positioning
    sepal_length_label.grid(row=0, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_length_input.grid(row=0, column=0, padx=10, pady=(10,7))
 
    sepal_width_label.grid(row=1, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_width_input.grid(row=1, column=0, padx=10, pady=(10,7))

    petal_length_label.grid(row=2, column=0, padx=10, pady=(10,7), sticky="e")
    petal_length_input.grid(row=2, column=0, padx=10, pady=(10,7))

    petal_width_label.grid(row=3, column=0, padx=10, pady=(10,7), sticky="e")
    petal_width_input.grid(row=3, column=0, padx=10, pady=(10,7))

    classity_button.grid(row=4, column=0, padx=(10,10), pady=10)

# Main Window + root
root=Tk()
root.geometry("600x450")
root['background']='MediumPurple'
root.title("❀ Iris Classification ❀")




# First Page 



# QUESTION 1
upload_label = Label(root, text="Please upload the data set :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')


upload_button =Button(root,text="Upload here",width=20 , font=('Georgia', 16),bg='#E6E6FA',fg='#111111',command=openFiles) 

upload_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
upload_button.grid(row=1, column=0, padx=10, pady=10)


# QUESTION 2
choose_model =Label(root, text="Please choose the model :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')

r=IntVar()
r.set("1")

svm_option=Radiobutton(root,          text="Support Vector Machine ",variable=r,value=1,width=30 , font=('Times New Roman', 16),bg='MediumPurple' )
bayes_option=Radiobutton(root,        text="Naive Bayes            ",variable=r         ,value=2, width=30 , font=('Times New Roman', 16),bg='MediumPurple')
decision_tree_option=Radiobutton(root,text="Decision Tree          ",variable=r,value=3,width=30 , font=('Times New Roman', 16),bg='MediumPurple')
   


# Training
train_button = Button(root, text="Train the model",width=30 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:check_uploaded_file(r.get()))

choose_model.grid(row=2, column=0, padx=10, pady=10, sticky="e")
svm_option.grid(row=3, column=0, padx=10, pady=10)
bayes_option.grid(row=4, column=0, padx=10, pady=10)
decision_tree_option.grid(row=5, column=0, padx=10, pady=10)
train_button.grid(row=6, column=0, padx=10, pady=10)

    

root.mainloop()
