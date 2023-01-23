from auxeticmop import load_pickled_dict_data
import numpy as np

result_parent = load_pickled_dict_data(r'C:\pythoncode\AuxeticMOP\abaqus data\FieldOutput_1')
result_offspring = load_pickled_dict_data(r'C:\pythoncode\AuxeticMOP\abaqus data\FieldOutput_offspring_1')
all_results = np.vstack((result_parent, result_offspring))
print(result_parent[1]['displacement']['yMax'][1])
