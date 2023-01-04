import GraphicUserInterface3 as GUI
import matplotlib.pyplot as plt
from time import sleep
import tkinter as tk
import pandas as pd
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

app = GUI.App()
# data1 = {'country': ['A', 'B', 'C', 'D', 'E'],
#          'gdp_per_capita': [45000, 42000, 52000, 49000, 47000]
#          }
# df1 = pd.DataFrame(data1)
#
# while True:
#     sleep(5)
#     app.show_canvas()
#     app.plot(0.5, 0.5)
#     sleep(5)

sleep(5)
while True:
    app.scatter(random.random(), random.random())
    print(app.setPath.get())
    sleep(1)