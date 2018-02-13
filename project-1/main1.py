from Tkinter import *

class GUI(Frame):

    def _init_(self,master=None):
        Frame._init_(self, master)
        self.grid()

        self.fnameLabel = Label(master, text="First Name")
        self.fnameLabel.grid()

        self.fnameEntry = StringVar()
        self.fnameEntry = Entry(textvariable=self.fnameEntry)
        self.fnameEntry.grid()

        self.lnameLabel = Label(master, text="Last Name")
        self.lnameLabel.grid()

        self.lnameEntry = StringVar()
        self.lnameEntry = Entry(textvariable=self.lnameEntry)
        self.lnameEntry.grid()

        def buttonClick():
            print("You pressed Submit!")
            print(self.fnameEntry.get() + " " + self.lnameEntry.get() +",you clicked the button!")

        self.submitButton = Button(master, text="Submit", command=buttonClick)
        self.submitButton.grid()

if __name__ == "__main__":
    guiFrame = GUI()    
    guiFrame.mainloop()

