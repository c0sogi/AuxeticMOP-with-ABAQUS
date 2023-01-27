from auxeticmop.FileIO import get_sorted_file_numbers_from_pattern
import os

os.chdir(r'C:\Users\dcas\Desktop\새 폴더 (2)\pickles')
print(get_sorted_file_numbers_from_pattern(r'FieldOutput_offspring_\d+'))