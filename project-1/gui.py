import numpy as np
import cv2
import Tkinter 
import Image, ImageTk

# Load an color image
img = cv2.imread('../../Desktop/House_sparrow04.jpg',1)

#Rearrang the color channel
b,g,r = cv2.split(img)
img = cv2.merge((r,g,b))
r, c = divmod(0, 100)
# A root window for displaying objects
root = Tkinter.Tk()  

# Convert the Image object into a TkPhoto object
im = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=im) 

# Put it in the display window
Tkinter.Label(root, image=imgtk).pack() 
Tkinter.grid(row=r,column=c)
root.mainloop() # Start the GUI



