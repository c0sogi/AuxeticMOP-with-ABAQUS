import GraphicUserInterface as GUI
import numpy as np
from time import sleep
import random
import pickle
from MutateAndValidate import visualize_one_cube

# app = GUI.App()
# sleep(5)
# while True:
#     app.scatter(0, random.random(), random.random())
#     app.plot(1, np.random.random(5), np.random.random(5))
#     print(app.setPath.get())
#     sleep(1)
with open(r'C:\Users\dcas\PythonCodes\Coop\pythoncode\23-1-2\data\args', mode='rb') as f:
    a = pickle.load(f)

visualize_one_cube(a['offspring'][0])