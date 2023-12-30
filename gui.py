from tkinter import *


root=Tk()
root.geometry("600x300")
root['background']='MediumPurple'
root.title("❀ Iris Classification ❀")

#Widgets

#Labels
sepal_length_label = Label(root, text="Sepal Length",width=20 , font=('Helvetica', 16),bg='MediumPurple')
sepal_width_label = Label(root, text="Sepal Width",width=20 , font=('Helvetica', 16),bg='MediumPurple')
petal_length_label = Label(root, text="Petal Length",width=20 , font=('Helvetica', 16),bg='MediumPurple')
petal_width_label = Label(root, text="Petal Width",width=20 , font=('Helvetica', 16),bg='MediumPurple')

#Inputs
sepal_length_input = Entry(root, width=20 , font=('Helvetica', 16))
sepal_width_input = Entry(root, width=20 , font=('Helvetica', 16))
petal_length_input = Entry(root, width=20 , font=('Helvetica', 16))
petal_width_input = Entry(root, width=20 , font=('Helvetica', 16))

#Buttons
classity_button = Button(root, text="Classify Now!",width=20 , font=('Helvetica', 12),bg='Purple', fg='white')


# Grid Positioning
sepal_length_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
sepal_length_input.grid(row=1, column=0, padx=10, pady=10)

sepal_width_label.grid(row=0, column=1, padx=10, pady=10, sticky="e")
sepal_width_input.grid(row=1, column=1, padx=10, pady=10)

petal_length_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
petal_length_input.grid(row=3, column=0, padx=10, pady=10)

petal_width_label.grid(row=2, column=1, padx=10, pady=10, sticky="e")
petal_width_input.grid(row=3, column=1, padx=10, pady=10)

classity_button.grid(row=4, column=0, padx=10, pady=10)


root.mainloop()
