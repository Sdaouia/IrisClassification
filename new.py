from tkinter import *
from tkinter import filedialog
import csv 
from tkinter import messagebox

global opened_file
opened_file=None

def openFiles():
    root.filename=filedialog.askopenfilename(initialdir="/",title="select file",filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    global opened_file
    opened_file=root.filename
    if root.filename:
        upload_button.config(text="Iris.csv",bg='#DCDCDC')
        # Read the content of the CSV file
        

def training(model_value,opened_file):

    print("Training with",model_value)
    with open(opened_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)
    #call the model training
    #go to page 2
    move_to_second_page()          
    

def check_uploaded_file(model_value):
      if(opened_file is not None):
                print("you entred your dataset successfully")
                training(model_value,opened_file)
      else:
          messagebox.showerror("Error", "Please enter your data set ! ")

def move_to_second_page():
      upload_label.grid_forget()
      upload_button.grid_forget()
      choose_model.grid_forget()
      svm_option.grid_forget()
      bayes_option.grid_forget()
      decision_tree_option.grid_forget()
      train_button.grid_forget()



# Main Window + root
root=Tk()
root.geometry("600x600")
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
