from tkinter import *

def hide_elements():
    #Hide labels, entries, and button
    sepal_length_input.grid_forget()
    sepal_length_label.grid_forget()
    sepal_width_input.grid_forget()
    sepal_width_label.grid_forget()
    petal_length_input.grid_forget()
    petal_length_label.grid_forget()
    petal_width_input.grid_forget()
    petal_width_label.grid_forget()
    classity_button.grid_forget()
    



root=Tk()
root.geometry("600x350")
root['background']='MediumPurple'
root.title("❀ Iris Classification ❀")

#Widgets

#Frame
Frame=LabelFrame(root,text="",bg="MediumPurple",padx=5,pady=10)
Frame.pack(padx=20,pady=20)



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
classity_button = Button(Frame, text="Classify Now!",command=hide_elements,width=20 , font=('Helvetica', 16),bg='Purple', fg='white')


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


root.mainloop()
