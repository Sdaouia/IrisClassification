from tkinter import *
from tkinter import filedialog
import csv 
from tkinter import messagebox

def on_submit(sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button):
    entry_values = [entry.get() for entry in [sepal_length_input,sepal_width_input,petal_length_input,petal_width_input]]
    try:
        float_values = [float(value) for value in entry_values]
        print("Float values:", float_values)
        forgetForme(sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button)
        move_to_third()
    except ValueError:
        messagebox.showerror("Error", "Please enter valid float numbers.")


global data
data=1
global model
model=1

def on_radio_button_selected(value):
    
    data = value
    # 1 -> Individual value
    # 2 -> Dataset
    print("type data=",data)

def on_model_selected(value):
    
    model = value
    # 1 -> Individual value
    # 2 -> Dataset
    print("model=",model)


def move_to_second(data,model):
    if(data==2):
        openFiles()
    else:
        openIrisForme()
   

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
    

def move_to_third():
    label=Label(Frame, text="Third Page",width=20 , font=('Helvetica', 16),bg='MediumPurple')
    label.pack()



def openFiles():
    root.filename=filedialog.askopenfilename(initialdir="/",title="select file",filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    if root.filename:
       
        # Read the content of the CSV file
        with open(root.filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)
        
                
        choose_label.grid_forget()
        individual_data_rb.grid_forget()
        csv_file_rb.grid_forget()
        choose_model.grid_forget()
        svm_option.grid_forget()
        bayes_option.grid_forget()
        decision_tree_option.grid_forget()
        next_button.grid_forget()
        move_to_third() 


def openIrisForme():
    choose_label.grid_forget()
    individual_data_rb.grid_forget()
    csv_file_rb.grid_forget()
    choose_model.grid_forget()
    svm_option.grid_forget()
    bayes_option.grid_forget()
    decision_tree_option.grid_forget()
    next_button.grid_forget()

    #Labels
    sepal_length_label = Label(Frame, text="Sepal Length",width=20 , font=('Helvetica', 16),bg='MediumPurple')
    sepal_length_label = Label(Frame, text="Sepal Length",width=20 , font=('Helvetica', 16),bg='MediumPurple')
    sepal_width_label = Label(Frame, text="Sepal Width",width=20 , font=('Helvetica', 16),bg='MediumPurple')
    petal_length_label = Label(Frame, text="Petal Length",width=20 , font=('Helvetica', 16),bg='MediumPurple')
    petal_width_label = Label(Frame, text="Petal Width",width=20 , font=('Helvetica', 16),bg='MediumPurple')

    #Inputs
    sepal_length_input = Entry(Frame, width=20 , font=('Helvetica', 16))
    #sepal_length_input.insert(0,"Entre sepal length :")

    sepal_width_input = Entry(Frame, width=20 , font=('Helvetica', 16))
    petal_length_input = Entry(Frame, width=20 , font=('Helvetica', 16))
    petal_width_input = Entry(Frame, width=20 , font=('Helvetica', 16))

    #Buttons
    classity_button = Button(Frame, text="Classify Now!",width=20 , font=('Helvetica', 16),bg='Purple', fg='white',command=lambda:on_submit(sepal_length_input,sepal_length_label,sepal_width_label,sepal_width_input,petal_length_input,petal_length_label,petal_width_label,petal_width_input,classity_button))
    

    # Grid Positioning
    sepal_length_label.grid(row=0, column=0, padx=10, pady=(10,7), sticky="e")
    sepal_length_input.grid(row=1, column=0, padx=10, pady=(10,7))
 
    sepal_width_label.grid(row=0, column=1, padx=10, pady=(10,7), sticky="e")
    sepal_width_input.grid(row=1, column=1, padx=10, pady=(10,7))

    petal_length_label.grid(row=2, column=0, padx=10, pady=(10,7), sticky="e")
    petal_length_input.grid(row=3, column=0, padx=10, pady=(10,7))

    petal_width_label.grid(row=2, column=1, padx=10, pady=(10,7), sticky="e")
    petal_width_input.grid(row=3, column=1, padx=10, pady=(10,7))

    classity_button.grid(row=4, column=0, padx=(10,10), pady=10)

    






# Main Window + Frame
root=Tk()
root.geometry("600x600")
root['background']='MediumPurple'
root.title("❀ Iris Classification ❀")

Frame=LabelFrame(root,text="",bg="MediumPurple",padx=5,pady=10)
Frame.pack(padx=20,pady=20)


# First Page 



# QUESTION 1
choose_label = Label(Frame, text="Do you want to entre :",width=40 , font=('Helvetica', 16),bg='white')

rb=IntVar()
rb.set("1")

individual_data_rb =Radiobutton(Frame,text="Individual Data",variable=rb,value=1,width=30 , font=('Helvetica', 16),bg='MediumPurple',command=lambda:on_radio_button_selected(rb.get()) ) 
csv_file_rb=Radiobutton(Frame,text="Upload CSV File",variable=rb,value=2,width=30 , font=('Helvetica', 16),bg='MediumPurple',command=lambda:on_radio_button_selected(rb.get()) )

choose_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
individual_data_rb.grid(row=1, column=0, padx=10, pady=10)
csv_file_rb.grid(row=2, column=0, padx=10, pady=10)



# QUESTION 2
choose_model =Label(Frame, text="Please choose the model :",width=40 , font=('Helvetica', 16),bg='white')

r=IntVar()
r.set("1")

svm_option=Radiobutton(Frame,text="Support Vector Machine",variable=r,value=1,width=30 , font=('Helvetica', 16),bg='MediumPurple',command=lambda:on_model_selected(r.get()) )
bayes_option=Radiobutton(Frame,text="Naive Bayes",variable=r         ,value=2, width=30 , font=('Helvetica', 16),bg='MediumPurple',command=lambda:on_model_selected(r.get()))
decision_tree_option=Radiobutton(Frame,text="Decision Tree",variable=r,value=3,width=30 , font=('Helvetica', 16),bg='MediumPurple',command=lambda:on_model_selected(r.get()))
   


# Move to next step
next_button = Button(Frame, text="NEXT",command=lambda:move_to_second(rb.get(),r.get()),width=30 , font=('Helvetica', 16),bg='Purple', fg='white')

choose_model.grid(row=3, column=0, padx=10, pady=10, sticky="e")
svm_option.grid(row=4, column=0, padx=10, pady=10)
bayes_option.grid(row=5, column=0, padx=10, pady=10)
decision_tree_option.grid(row=6, column=0, padx=10, pady=10)
next_button.grid(row=7, column=0, padx=10, pady=10)

    

root.mainloop()
