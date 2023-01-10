import os
from main import plot_previous_data, make_and_start_process, remove_file, one_generation
from GraphicUserInterface import App
from PostProcessing import Mop

if __name__ == '__main__':
    # Make an interface and receive parameters
    gui_process, parent_conn, child_conn = make_and_start_process(target=App, duplex=True, daemon=True)
    set_path, parameters = parent_conn.recv()
    parameters.post_initialize()
    os.chdir(set_path)

    # load previous plotting data
    remove_file(file_name='args')
    mop_for_plot = Mop(conn_to_gui=parent_conn)
    plot_previous_data(conn_to_gui=parent_conn, use_manual_rp=False)

    # Start working
    one_generation(w=1, restart=False, params=parameters, mop=mop_for_plot)
