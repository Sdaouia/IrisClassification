from tkinter import *
from tkinter import filedialog
import csv 
from tkinter import messagebox
from PIL import ImageTk,Image
import pandas as pd
from naiveBayes import nbModel , nbIrisClassifier 
from SVM_model import SVM ,SVM_classification
import os


global opened_file
opened_file=None   # before opening any file it' intialised with None



def openFiles():
    #open the file system and allow the user to select a csv file
    root.filename=filedialog.askopenfilename(initialdir="/",title="select file",filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    global opened_file
    opened_file=root.filename
    if root.filename:
        name_file=os.path.basename(root.filename)
        upload_button.config(text=name_file,bg='#DCDCDC') # update the text of <upload button> to the opened file name
        
    


def training(model_value,opened_file):
    #call the functions to train the model 
    if(model_value==1):        #if model value = 1 it's an svm model
        result=SVM(opened_file)    
        move_to_second_page(result,model_value) #we move to second page

    if(model_value==2):        #if model value = 2 it's a naive bayes model
        result=nbModel(opened_file)
        move_to_second_page(result,model_value) #we move to second page




def check_uploaded_file(model_value):
      #if the user select a file we train the data set else we show an error message
      if(opened_file is not None and model_value!=3):
                training(model_value,opened_file)
      if(opened_file==None):
          messagebox.showerror("Error", "Please upload iris data set ! ")
      if(opened_file and model_value==3): #if the user select decision tree model we show error message
          messagebox.showerror("Error", " << Decision Tree Model >> is not available ,Please Try with another model ! ")         
    


def on_submit(forme_label,sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button,model_value):
    #before the classification of the instance we check if the user has entred valid float numbers
    entry_values = [entry.get() for entry in [sepal_length_input,sepal_width_input,petal_length_input,petal_width_input]]
    try:
        float_values = [float(value) for value in entry_values]
        forgetForme(forme_label,sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button)

        #call classification function using the trained model 
        if(model_value==1):
          variety=SVM_classification(float(sepal_length_input.get()), float(sepal_width_input.get()), float(petal_length_input.get()), float(petal_width_input.get()))
          move_to_forth_page(model_value,variety)

        if(model_value==2):
          variety=nbIrisClassifier(float(sepal_length_input.get()), float(sepal_width_input.get()), float(petal_length_input.get()), float(petal_width_input.get()))
          move_to_forth_page(model_value,variety)
 
    except ValueError:
        messagebox.showerror("Error", "Please enter valid float numbers.")



def move_to_second_page(result,model_value):
      #remove the previous page
      Welcome_label.grid_forget()
      upload_label.grid_forget()
      upload_button.grid_forget()
      choose_model.grid_forget()
      svm_option.grid_forget()
      bayes_option.grid_forget()
      decision_tree_option.grid_forget()
      train_button.grid_forget()
      
      #svm and naive bayes model
      if(model_value==1):
          model="SVM"
      else:    
          model="Naive Bayes"


      # create the second page
      #this page show details about the trained model with iris data set such as accuracy and classification report
      
      accuracy_label=Label(root, text="The accuracy :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')

      accuracy=Label(root, text=f"The accuracy of the {model} model is {result['accuracy']:.2f} .",width=50 , font=('Georgia', 13),fg='black',bg='MediumPurple',anchor='w')
      accuracy_definition=Label(root, text=f"This means that it can correctly predict the type of the Iris flower {round(result['accuracy'] * 100)} % of the time.",width=65 , font=('Georgia', 13),fg='black',bg='MediumPurple',anchor='w')

      classification_report_label=Label(root, text="Classification report :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')

      classification_report=Label(root, text=result['classification_report'],width=50 , font=('Georgia', 13),fg='black',bg='MediumPurple',anchor='w')

      confusion_matrix_label=Label(root, text="Confusion matrix :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')
      
      confusion_matrix=Label(root, text=pd.DataFrame(result['confusion_matrix'], columns=['Setosa', 'Versicolor', 'Virginica'], index=['Setosa', 'Versicolor', 'Virginica']),width=50 , font=('Georgia', 13),fg='black',bg='MediumPurple',anchor='w')

      instance_button=Button(root, text="Enter an instance",width=30 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:move_to_third_page(accuracy_label,accuracy,accuracy_definition,classification_report_label,classification_report,confusion_matrix_label,confusion_matrix,instance_button,model_value))

      accuracy_label.grid(row=0, column=0, padx=20, pady=5, sticky="e")
      accuracy.grid(row=1, column=0, padx=20, pady=5, sticky="e")
      accuracy_definition.grid(row=2, column=0, padx=20, pady=5, sticky="e")
      classification_report_label.grid(row=3, column=0, padx=20, pady=5, sticky="e")
      classification_report.grid(row=4, column=0, padx=20, pady=5, sticky="e")
      confusion_matrix_label.grid(row=5, column=0, padx=20, pady=5, sticky="e")
      confusion_matrix.grid(row=6, column=0, padx=20, pady=5, sticky="e")
      instance_button.grid(row=7, column=0, padx=20, pady=10)



def move_to_third_page(accuracy_label,accuracy,accuracy_definition,classification_report_label,classification_report,confusion_matrix_label,confusion_matrix,instance_button,model_value):

    #remove the previous page 
    accuracy.grid_forget()
    accuracy_label.grid_forget()
    accuracy_definition.grid_forget()
    classification_report_label.grid_forget()
    confusion_matrix_label.grid_forget()
    classification_report.grid_forget()
    confusion_matrix.grid_forget()
    instance_button.grid_forget()
    
    #create the third page 
    #Labels
    forme_label = Label(root, text="Entre the instance here !",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    sepal_length_label = Label(root, text="Sepal Length",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    sepal_width_label = Label(root, text="Sepal Width",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    petal_length_label = Label(root, text="Petal Length",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    petal_width_label = Label(root, text="Petal Width",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')

    #Inputs
    sepal_length_input = Entry(root, width=30 , font=('Helve', 16))
    

    sepal_width_input = Entry(root, width=30 , font=('Helvetica', 16))
    petal_length_input = Entry(root, width=30 , font=('Helvetica', 16))
    petal_width_input = Entry(root, width=30 , font=('Helvetica', 16))

    #Buttons
    classity_button = Button(root, text="Classify Now!",width=30 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:on_submit(forme_label,sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button,model_value))
    

  
      # Grid Positioning
    forme_label.grid(row=0, column=0, padx=10, pady=20)
    sepal_length_label.grid(row=1, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_length_input.grid(row=1, column=0, padx=10, pady=(10,7))
 
    sepal_width_label.grid(row=2, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_width_input.grid(row=2, column=0, padx=10, pady=(10,7))

    petal_length_label.grid(row=3, column=0, padx=10, pady=(10,7), sticky="e")
    petal_length_input.grid(row=3, column=0, padx=10, pady=(10,7))

    petal_width_label.grid(row=4, column=0, padx=10, pady=(10,7), sticky="e")
    petal_width_input.grid(row=4, column=0, padx=10, pady=(10,7))

    classity_button.grid(row=5, column=0, padx=(10,10), pady=50)




def forgetForme(forme_label,sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button):
    #remove the third page (the forme)
    forme_label.grid_forget()
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
     
     #create the forth page 

     variety_label = Label(root, text=f"The instance is {variety}",width=40 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
     image_label =Label(root)
     try_again_button=Button(root, text="Try a new instance",width=20 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:come_back_to_third(variety_label,image_label,try_again_button,model_value))
     

     if(variety=="Setosa"):
       img = Image.open("setosa.jpg")
       resized_img = img.resize((400, 400))
       photo = ImageTk.PhotoImage(resized_img)
       image_label.config(image=photo)
       image_label.image = photo 
    
     if(variety=="Virginica"):
        img = Image.open("virginica.jpg")
        resized_img = img.resize((400, 400))
        photo = ImageTk.PhotoImage(resized_img)
        image_label.config(image=photo)
        image_label.image = photo 

     if(variety=="Versicolor"):
        img = Image.open("versicolore.jpg")
        resized_img = img.resize((400, 400))
        photo = ImageTk.PhotoImage(resized_img)
        image_label.config(image=photo)
        image_label.image = photo
    
     variety_label.grid(row=0, column=0, padx=20, pady=10)
     image_label.grid(row=1, column=0, padx=20, pady=10,sticky="e")
     try_again_button.grid(row=2, column=0, padx=10, pady=10)
     


def come_back_to_third(variety_label,image_label,try_again_button,model_value):
    
    #if the user want to try another instance we back to the forme
    variety_label.grid_forget()
    image_label.grid_forget()
    try_again_button.grid_forget()

    #Labels


    forme_label = Label(root, text="Entre the instance here !",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    sepal_length_label = Label(root, text="Sepal Length",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    sepal_width_label = Label(root, text="Sepal Width",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    petal_length_label = Label(root, text="Petal Length",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')
    petal_width_label = Label(root, text="Petal Width",width=50 , font=('Georgia', 16,'bold'),bg='MediumPurple',fg='white',anchor='w')

    #Inputs
    sepal_length_input = Entry(root, width=30 , font=('Helve', 16))
    

    sepal_width_input = Entry(root, width=30 , font=('Helvetica', 16))
    petal_length_input = Entry(root, width=30 , font=('Helvetica', 16))
    petal_width_input = Entry(root, width=30 , font=('Helvetica', 16))

    #Buttons
    classity_button = Button(root, text="Classify Now!",width=30 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:on_submit(forme_label,sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button,model_value))
    

  
      # Grid Positioning
    forme_label.grid(row=0, column=0, padx=10, pady=20)
    sepal_length_label.grid(row=1, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_length_input.grid(row=1, column=0, padx=10, pady=(10,7))
 
    sepal_width_label.grid(row=2, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_width_input.grid(row=2, column=0, padx=10, pady=(10,7))

    petal_length_label.grid(row=3, column=0, padx=10, pady=(10,7), sticky="e")
    petal_length_input.grid(row=3, column=0, padx=10, pady=(10,7))

    petal_width_label.grid(row=4, column=0, padx=10, pady=(10,7), sticky="e")
    petal_width_input.grid(row=4, column=0, padx=10, pady=(10,7))

    classity_button.grid(row=5, column=0, padx=(10,10), pady=50)
    

# Main Window 
root=Tk()
root.geometry("750x570")
root['background']='MediumPurple'
root.title("❀ Iris Classification ❀")




# First Page 

# WELCOME
Welcome_label = Label(root, text="Welcome To Iris Classification",width=40 , font=('Georgia', 20,'bold'),fg='#6A0DAD',bg='MediumPurple')



# QUESTION 1
upload_label = Label(root, text="Please upload the Iris data set :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')
upload_button =Button(root,text="Upload here",width=20 , font=('Georgia', 16),bg='#E6E6FA',fg='#111111',command=openFiles) 

Welcome_label.grid(row=0, column=0, padx=10, pady=20)
upload_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
upload_button.grid(row=2, column=0, padx=10, pady=10)


# QUESTION 2
choose_model =Label(root, text="Please choose the model :",width=40 , font=('Georgia', 16,'bold'),fg='white',bg='MediumPurple',anchor='w')

r=IntVar()
r.set("1")

svm_option=Radiobutton(root          ,text="Support Vector Machine        ",variable=r,value=1,width=30 , font=('Times New Roman', 16),bg='MediumPurple' )
bayes_option=Radiobutton(root        ,text="Naive Bayes                   ",variable=r,value=2, width=30 , font=('Times New Roman', 16),bg='MediumPurple')
decision_tree_option=Radiobutton(root,text="Decision Tree (Available Soon)",variable=r,value=3,width=30 , font=('Times New Roman', 16),bg='MediumPurple')
   


# Training
train_button = Button(root, text="Train the model",width=30 , font=('Georgia', 16),bg='#6A0DAD', fg='white',command=lambda:check_uploaded_file(r.get()))

choose_model.grid(row=3, column=0, padx=10, pady=10, sticky="e")
svm_option.grid(row=4, column=0, padx=10, pady=10)
bayes_option.grid(row=5, column=0, padx=10, pady=10)
decision_tree_option.grid(row=6, column=0, padx=10, pady=10)
train_button.grid(row=7, column=0, padx=10, pady=10)

    

root.mainloop()
