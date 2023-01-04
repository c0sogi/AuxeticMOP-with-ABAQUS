# AuxeticMOP
This code is currently in development!
I am using Abaqus CAE software to research the optimal properties of an Auxetic cell.
These codes are used to generate data about 3D cell topology to be input into Abaqus, and to derive results by running it with Abaqus.

Firstly, In main.py, main process makes a gui_process, parent_conn, child_conn by calling class App from GraphicUserInterface.py.
The class App is what we are going to put parameters into main process & abaqus, and retreive plotting data from results(pareto front of fitness values, hypervolume).
Both parent_conn and child_conn are the Pipe object that connect between main process and gui process.
From parent_conn, the parent process (main) gets the set_path parameter from gui process, thereby we can save the parameters from gui into a file'PARAMS_MAIN', which will be loaded into main process.
Removing file 'args' is needed because 'args' is a file that contains commands sent to abaqus by the main process. 
If abaqus terminates abnormally, this file may not be deleted from Abaqus, and Abaqus may arbitrarily proceed with tasks without commands from the main process.
Nextly, The Abaqus CAE process will open. The abaqus can be opened with Non-GUI mode(noGUI), and GUI mode(script). You can choose it from gui.
Abaqus will evaluate its mechanical property of these topologies, which were created from existing parents, using 'generate_offspring' function.
Finally, the fitness values will be calculated and visualized to gui, and iterates previous steps until 'gen_idx' hits 'end_gen'.
