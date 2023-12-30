from tkinter import *
from tkinter import filedialog
import csv 

global data
global model

def openFiles():
    root.filename=filedialog.askopenfilename(initialdir="/",title="select file",filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    if root.filename:
        from_1_to_2()
        # Read the content of the CSV file
        with open(root.filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)





def from_1_to_2():
    choose_label.grid_forget()
    individual_data_button.grid_forget()
    csv_file_button.grid_forget()

    r=IntVar()
    r.set("1")
    global choose_model
    global svm_option
    global bayes_option
    global decision_tree_option
    global next_button

    choose_model =Label(Frame, text="Please choose the model :",width=40 , font=('Helvetica', 16),bg='MediumPurple')

    svm_option=Radiobutton(Frame,text="Support Vector Machine",variable=r,value=1,width=30 , font=('Helvetica', 16),bg='MediumPurple' )
    bayes_option=Radiobutton(Frame,text="Naive Bayes",variable=r         ,value=2, width=30 , font=('Helvetica', 16),bg='MediumPurple')
    decision_tree_option=Radiobutton(Frame,text="Decision Tree",variable=r,value=3,width=30 , font=('Helvetica', 16),bg='MediumPurple')
   
    next_button = Button(Frame, text="NEXT",width=30,command=from_2_to_3 , font=('Helvetica', 16),bg='Purple', fg='white')


    choose_model.grid(row=0, column=0, padx=10, pady=10, sticky="e")
    svm_option.grid(row=1, column=0, padx=10, pady=10)
    bayes_option.grid(row=2, column=0, padx=10, pady=10)
    decision_tree_option.grid(row=3, column=0, padx=10, pady=10)
    next_button.grid(row=4, column=0, padx=10, pady=10)

    
def from_2_to_3():
    choose_model.grid_forget()
    svm_option.grid_forget()
    bayes_option.grid_forget()
    decision_tree_option.grid_forget()
    next_button.grid_forget()




root=Tk()
root.geometry("600x350")
root['background']='MediumPurple'
root.title("❀ Iris Classification ❀")

Frame=LabelFrame(root,text="",bg="MediumPurple",padx=5,pady=10)
Frame.pack(padx=20,pady=20)


# First Page 
choose_label = Label(Frame, text="Do you want to entre :",width=40 , font=('Helvetica', 16),bg='MediumPurple')
individual_data_button = Button(Frame, text="Individual Data",command=from_1_to_2,width=30 , font=('Helvetica', 16),bg='Purple', fg='white')
csv_file_button = Button(Frame, text="Upload CSV File",command=openFiles,width=30 , font=('Helvetica', 16),bg='Purple', fg='white')

choose_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
individual_data_button.grid(row=1, column=0, padx=10, pady=10)
csv_file_button.grid(row=2, column=0, padx=10, pady=10)

root.mainloop()
