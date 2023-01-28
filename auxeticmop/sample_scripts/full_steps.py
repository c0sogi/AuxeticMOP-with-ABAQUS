"""
Run full GA steps
"""


def run():
    import os
    from ..GraphicUserInterface import App, Visualizer, plot_previously_plotted_data
    from ..GeneticAlgorithm import NSGAModel
    from ..Network import Server, make_and_start_process, start_abaqus_cae
    from ..ParameterDefinitions import material_property_definitions

    HOST = 'localhost'
    PORT = 12345
    # Open socket server
    server = Server(host=HOST, port=PORT, option='json', run_nonblocking=True)

    try:
        # Make an interface and receive parameters
        gui_process, parent_conn, child_conn = make_and_start_process(target=App, duplex=True, daemon=True)
        set_path, parameters = parent_conn.recv()
        parameters.post_initialize()
        os.chdir(set_path)

        # Start ABAQUS
        abaqus_process = start_abaqus_cae(option=parameters.abaqus_mode)

        # Load previous plotting data
        visualizer = Visualizer(conn_to_gui=parent_conn)
        plot_previously_plotted_data(visualizer=visualizer, params=parameters)

        # Run GA process
        ga_model = NSGAModel(params=parameters, material_properties=material_property_definitions,
                             visualizer=visualizer, random_topology_density=0.3)
        ga_model.run(server=server)
    except Exception as e:
        print(e)
    finally:
        server.close()


if __name__ == '__main__':
    run()
