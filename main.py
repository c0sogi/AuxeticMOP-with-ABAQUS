import os
from time import sleep
from auxeticmop.GraphicUserInterface import App, Visualizer, load_previous_data
from auxeticmop.GeneticAlgorithm import NSGAModel
from auxeticmop.Network import Server, make_and_start_process, start_abaqus_cae
from auxeticmop.ParameterDefinitions import material_property_definitions


if __name__ == '__main__':
    # Open socket server
    server = Server(host='localhost', port=12345, option='json', run_nonblocking=True)

    # Make an interface and receive parameters
    gui_process, parent_conn, child_conn = make_and_start_process(target=App, duplex=True, daemon=True)
    set_path, parameters = parent_conn.recv()
    parameters.post_initialize()
    os.chdir(set_path)

    # Start ABAQUS and wait for socket connection
    abaqus_process = start_abaqus_cae(script_name=parameters.abaqus_script, option=parameters.abaqus_mode)
    print('Connecting socket to ABAQUS ...')
    while len(server.connected_clients) == 0:
        sleep(1.0)

    # Load previous plotting data
    visualizer = Visualizer(conn_to_gui=parent_conn)
    load_previous_data(visualizer=visualizer, params=parameters)

    # Run GA process
    ga_model = NSGAModel(params=parameters, material_properties=material_property_definitions,
                         visualizer=visualizer, random_topology_density=0.5)
    ga_model.run(server=server)
