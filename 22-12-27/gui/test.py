import tkinter as tk
my_w = tk.Tk()
my_w.geometry("500x500")  # Size of the window

def my_upd():
    print('Radiobutton  value :',r1_v.get())

r1_v = tk.StringVar()  # We used string variable here
r1_v.set(None)     # Can assign value Appear or Failed

r1 = tk.Radiobutton(my_w, text='Passed', variable=r1_v, value='Passed',command=my_upd)
r1.grid(row=1,column=1)

r2 = tk.Radiobutton(my_w, text='Failed', variable=r1_v, value='Failed',command=my_upd)
r2.grid(row=1,column=2)

r3 = tk.Radiobutton(my_w, text='Appearing', variable=r1_v, value='Appear',command=my_upd )
r3.grid(row=1,column=3)

my_w.mainloop()  # Keep the window open