"""
Run full GA steps
"""


def run():
    """
    <!> Caution
    You must use if __name__ == '__main__' for running GA,
    If __name__=='__main__' is not used, multiple servers are running and multiple bindings occur on the same port,
    which causes OSError: [WinError 10048] server-binding error. The reason this phenomenon occurs is as follows.

    [1] Some functions like make_and_start_process() and 'start_abaqus_cae() creates new process.
    [2] If creating process using multiprocessing.Process(), that process will run the code at the top level of the script.
    [3] If you have code that you only want to run when the script is run directly, you can put it in the block of code
    indented under if __name__ == '__main__':, so that it will not be executed when the script is imported as a module and
    run in a new process. Without that condition, the same block of code will run multiple times in multiple processes
    which can lead to unexpected results.
    """
    import os
    from ..GraphicUserInterface import App, Visualizer, plot_previously_plotted_data
    from ..GeneticAlgorithm import NSGAModel
    from ..Network import Server, make_and_start_process, start_abaqus_cae
    from ..ParameterDefinitions import material_property_definitions, fitness_definitions

    HOST = 'localhost'
    PORT = 12345

    # Open socket server
    server = Server(host=HOST, port=PORT, option='json', run_nonblocking=True)

    # Make an interface and receive parameters
    gui_process, parent_conn, child_conn = make_and_start_process(target=App, duplex=True, daemon=True)
    set_path, parameters = parent_conn.recv()
    parameters.post_initialize()
    os.chdir(set_path)

    # Start ABAQUS
    abaqus_process = start_abaqus_cae()

    # Load previous plotting data
    visualizer = Visualizer(conn_to_gui=parent_conn)
    plot_previously_plotted_data(visualizer=visualizer, params=parameters)

    try:
        # Run GA process
        ga_model = NSGAModel(params=parameters, material_properties=material_property_definitions, visualizer=visualizer,
                             fitness_definitions=fitness_definitions,  random_topology_density=0.3)
        ga_model.evolve(server=server)
    except Exception as e:
        print(e)
    finally:
        server.close()
        abaqus_process.kill()


if __name__ == '__main__':
    run()
