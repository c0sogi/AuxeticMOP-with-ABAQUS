import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter import messagebox
import os
import pickle
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
from typing import TYPE_CHECKING
from PostProcessing import get_datum_hv, get_hv_from_datum_hv, find_pareto_front_points, evaluate_fitness_values
from FileIO import parent_import


@dataclass  # Use @dataclass(kw_only=True) for Python version >= 3.10
class Parameters:
    abaqus_script_name: str = 'ABQ.py'  # abaqus python script filename e.g., ABQ.py
    abaqus_execution_mode: str = 'script'  # noGUI: without abaqus gui, script: with abaqus gui
    mode: str = 'GA'  # GA mode
    evaluation_version: str = 'ver3'  # fitness value evaluation mode
    restart_pop: int = 0  # 0 for no-restart, 1~ for initial restarting population
    ini_pop: int = 1  # First population number, default: 1
    end_pop: int = 100  # Last population number
    ini_gen: int = 1  # First generation number, default: 1
    end_gen: int = 50  # Last generation number
    mutation_rate: float = 0.1  # mutation process option
    unit_l: float = 3  # Voxel length
    lx: int = 10  # Number of voxels in x-direction
    ly: int = 10  # Number of voxels in y-direction
    lz: int = 10  # Number of voxels in z-direction
    divide_number: int = 1  # up-scaling factor
    mesh_size: float = 0.25  # abaqus meshing option
    dis_y: float = -0.005  # abaqus boundary condition option
    material_modulus: float = 1100  # abaqus material property option
    poissons_ratio: float = 0.4  # abaqus material property option
    density: float = 1  # abaqus material property option
    MaxRF22: float = 0.01  # fitness value evaluation option
    penalty_coefficient: float = 0.1  # fitness value evaluation option
    sigma: float = 1  # filtering option
    threshold: float = 0.5  # filtering option
    n_cpus: int = 16  # abaqus option
    n_gpus: int = 0  # abaqus option
    timeout: float = 0.5  # validation process option

    def post_initialize(self):  # This function is only used in main.py, setting initial values to real value to be used
        self.lx *= self.divide_number
        self.ly *= self.divide_number
        self.lz *= self.divide_number  # number of voxels after increasing resolution
        self.unit_l /= self.divide_number
        unit_lx_total = self.lx * self.unit_l
        unit_ly_total = self.ly * self.unit_l
        unit_lz_total = self.lz * self.unit_l
        self.mesh_size *= self.unit_l
        self.dis_y *= unit_ly_total  # boundary condition (displacement)
        self.MaxRF22 *= unit_lx_total * unit_lz_total * self.material_modulus  # 0.01 is strain


# Define parameters for gui
PARAMETER_FILE_NAME = '_PARAMETERS_'
padx = 5  # Padding width
pady = 5  # Padding height
left_width = 1400  # default width: 400
right_width = 400
height = 750
button_width = 15
parameters = Parameters()
parameters_dict = asdict(parameters)


class App:  # GUI class
    def __init__(self, conn=None):
        # Root configuration
        self.conn = conn
        self.ready_to_run = False
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback_quit)
        self.root.title('Abaqus-Python Unified Control Interface')
        self.root.config(background='#FFFFFF')
        self.root.resizable(False, False)

        # Frames
        self.up_frame = tk.Frame(self.root, width=left_width + right_width + 2 * padx, height=100)
        self.up_frame.config(background='#FFFFFF')
        self.up_frame.grid(row=0, column=0, padx=padx, pady=pady / 2)
        self.up_frame.grid_propagate(False)
        self.down_frame = tk.Frame(self.root, width=right_width + left_width + 2 * padx, height=height)
        self.down_frame.config(background='#FFFFFF')
        self.left_frame = tk.Frame(self.down_frame, width=left_width, height=height, padx=padx, pady=pady)
        self.right_frame = tk.Frame(self.down_frame, width=right_width, height=height, padx=padx, pady=pady)

        # Elements
        self.set_path_title = tk.Label(self.up_frame,
                                       text='Choose the folder containing the Abaqus script and CSV files.')
        self.set_path_title.config(background='#FFFFFF')
        self.set_path_display = tk.Listbox(self.up_frame, width=50, height=1)
        self.set_path_btn = tk.Button(self.up_frame, text='Browse folder...', width=button_width,
                                      command=self.onclick_set_path_button)
        self.submit_btn = tk.Button(self.right_frame, width=button_width, text='Save presets',
                                    command=self.onclick_submit_btn)
        self.exit_btn = tk.Button(self.right_frame, width=button_width, text='Exit',
                                  command=self.onclick_exit_btn, background='#FF0000', foreground='#FFFFFF')
        self.set_default_btn = tk.Button(self.right_frame, width=button_width, text='Load defaults',
                                         command=self.onclick_set_default_btn)
        self.setPath = tk.StringVar()
        self.string_vars = [tk.StringVar() for _ in Parameters.__dict__]
        self.figure, self.ax = plt.subplots(nrows=1, ncols=2,
                                            figsize=((left_width - padx) / 100, (height - pady) / 100), dpi=100)
        self.bar = FigureCanvasTkAgg(self.figure, self.left_frame)
        self.bar.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

        # Show main
        self.set_path_title.pack()
        self.set_path_display.pack()
        self.set_path_btn.pack()
        self.root.mainloop()

    def callback_quit(self):
        self.root.quit()

    def show_canvas(self):
        # Show graphs to be plotted
        self.ax[0].set(title='Pareto Fronts', xlabel='Objective function 1', ylabel='Objective function 2')
        self.ax[1].set(title='Hyper Volume by Generation', xlabel='Generation', ylabel='Hyper volume')
        self.ax[0].grid(True)
        self.ax[1].grid(True)
        self.ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        if self.conn is not None:
            self.update_canvas()

    def update_canvas(self, polling_rate: float = 10.0):
        if self.conn.poll():  # Checking if any received data from main.py for every 1/polling_rate second
            try:
                _x1, _y1, _x2y2 = self.conn.recv()
                _x2, _y2 = zip(*_x2y2.items())
                print('[GUI] Received: ', _x1, _y1, _x2, _y2)

                # Randomize plotting options to make data more noticeable
                _color_1, _color_2 = np.random.rand(3, ), np.random.rand(3, )
                _plot_options = {
                    'marker': np.random.choice(
                        ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')),
                    'color': _color_1,
                    'markeredgecolor': _color_1,
                    'markerfacecolor': _color_2,
                    'markersize': 8,
                    'markeredgewidth': 2}
                self.ax[0].plot(_x1, _y1, **_plot_options)
                self.ax[1].clear()
                for _generation, _hv, _line in zip(_x2, _y2, self.ax[0].lines):
                    self.ax[1].scatter(_generation, _hv, marker=_line.get_marker(), c=[_line.get_markerfacecolor()],
                                       edgecolors=_line.get_markeredgecolor(), s=_line.get_markersize() ** 2,
                                       linewidth=_line.get_markeredgewidth())
                self.ax[0].grid(True)
                self.ax[1].grid(True)
                self.bar.draw()
            except Exception as error_message:
                print('[GUI] An plotting error occurred:', error_message)
        self.root.after(int(1000 / polling_rate), self.update_canvas)

    def onclick_set_path_button(self):
        try:
            _folder_path = askdirectory(initialdir="./")
            if _folder_path:
                self.set_path_display.delete(0, "end")
                self.set_path_display.insert(0, _folder_path)
            self.setPath.set(self.set_path_display.get(0))
            os.chdir(self.setPath.get())

            _params_already_exists = True if os.path.isfile(PARAMETER_FILE_NAME) else False
            if _params_already_exists:
                with open(PARAMETER_FILE_NAME, mode='rb') as f_params:
                    self.set_path_display.config(background='#00FF00')
                    self.set_path_title.config(text='Preset parameter value found.')
                    self.show_parameters(loaded=pickle.load(f_params))
                    self.ready_to_run = False
            else:
                self.set_path_display.config(background='#FF0000')
                self.set_path_title.config(text='Parameter not found. Please set below.')
                self.show_parameters(loaded=None)
                self.ready_to_run = False

        except Exception as error_message:
            messagebox.showerror("Error", f"An error occurred:\n{error_message}")

    def onclick_set_default_btn(self):
        for _idx, _value in enumerate(parameters_dict.values()):
            self.string_vars[_idx].set(_value)

    def return_radiobutton_frame_instead_of_entry(self, key: str, str_var_idx: int, name_dict: dict):  # right_Frame
        _radiobutton_frame = tk.Frame(self.right_frame, width=left_width - 2 * padx,
                                      height=height / len(parameters_dict) - pady)
        if len(name_dict[key]) == 2:
            _label_width = 25
            _radio_button_width = 9
        elif len(name_dict[key]) == 3:
            _label_width = 24
            _radio_button_width = 5
        else:
            raise ValueError('Please make sure the number of checkboxes is less than 3.')
        _label = tk.Label(_radiobutton_frame, width=_label_width, text=translator(s=key, flip=False), anchor='w')
        _label.grid(row=0, column=0)

        for menu_idx, menu in enumerate(name_dict[key]):
            _radio_button = tk.Radiobutton(_radiobutton_frame, text=menu, variable=self.string_vars[str_var_idx],
                                           width=_radio_button_width, value=menu, tristatevalue=menu)
            _radio_button.grid(row=0, column=menu_idx + 1)
        return _radiobutton_frame

    def onclick_submit_btn(self):
        for params_idx, key in enumerate(parameters_dict.keys()):
            parameters_dict[key] = atoi(self.string_vars[params_idx].get())
        if self.ready_to_run and self.conn is not None:
            self.conn.send((self.setPath.get(), Parameters(**parameters_dict)))
            self.submit_btn.config(background='#0000FF', foreground='#FFFFFF', text='Running...')
        else:
            with open(PARAMETER_FILE_NAME, mode='wb') as f_params:
                pickle.dump(parameters_dict, f_params)
                print(f'[GUI] Dumping to "{os.getcwd()}" Complete')
                for key, value in parameters_dict.items():
                    print(f'- {key}: {value}')
            self.submit_btn.config(background='#00FF00', text='Run')
            self.ready_to_run = True

    def onclick_exit_btn(self):
        print('[GUI] Closing GUI...')
        self.root.quit()

    def show_parameters(self, loaded: None | dict) -> None:
        radiobutton_name_dict = {
            'abaqus_execution_mode': ('noGUI', 'script'),
            'mode': ('GA', 'random'),
            'evaluation_version': ('ver1', 'ver2', 'ver3')
        }
        self.down_frame.grid(row=1, column=0, padx=padx, pady=pady / 2)
        self.down_frame.grid_propagate(False)
        self.left_frame.grid(row=1, column=0, padx=padx, pady=pady / 2)
        self.left_frame.grid_propagate(False)
        self.right_frame.grid(row=1, column=1, padx=padx, pady=pady / 2)
        self.right_frame.grid_propagate(False)
        for row_idx, key in enumerate(asdict(Parameters()).keys()):
            if key in radiobutton_name_dict.keys():
                _radio_button_frame = self.return_radiobutton_frame_instead_of_entry(key=key, str_var_idx=row_idx,
                                                                                     name_dict=radiobutton_name_dict)
                _radio_button_frame.grid(row=row_idx, column=0)

            else:
                _entry_frame = tk.Frame(self.right_frame, width=left_width - 2 * padx,
                                        height=height / len(asdict(Parameters()).keys()) - pady)
                _label = tk.Label(_entry_frame, width=25, text=translator(s=key, flip=False), anchor='w')
                _entry = tk.Entry(_entry_frame, width=25, textvariable=self.string_vars[row_idx])
                _entry_frame.grid(row=row_idx, column=0)
                _label.grid(row=0, column=0)
                _entry.grid(row=0, column=1)
            self.string_vars[row_idx].set('')

        empty_label = tk.Label(self.right_frame)
        empty_label.grid(row=len(asdict(Parameters()).keys()))
        self.submit_btn.grid(row=len(asdict(Parameters()).keys()) + 1)
        self.exit_btn.grid(row=len(asdict(Parameters()).keys()) + 2)
        self.set_default_btn.grid(row=len(asdict(Parameters()).keys()) + 3)

        if loaded:
            for params_idx, (key, value) in enumerate(loaded.items()):
                self.string_vars[params_idx].set(value)
        self.show_canvas()


def atoi(s: str) -> str | int | float:
    try:
        float(s)
        # s is integer or float type
        try:
            int(s)
            # s is int type
            return int(s)
        except ValueError:
            # s is float type
            return float(s)
    except ValueError:
        # s is not number, just string
        return s


def translator(s: str, flip: bool) -> str:
    dictionary = {'abaqus_script_name': 'Filename of ABAQUS script',
                  'abaqus_execution_mode': 'ABAQUS execution mode',
                  'mode': 'GA Mode',
                  'evaluation_version': 'GA evaluation version',
                  'restart_pop': '[P] Restart from population',
                  'ini_pop': '[P] First Population',
                  'end_pop': '[P] Last Population',
                  'ini_gen': '[G] First Generation',
                  'end_gen': '[G] Last Generation',
                  'mutation_rate': 'Mutation rate(0~1)',
                  'unit_l': 'Voxel unit length(mm)',
                  'lx': 'Voxel number in X-direction',
                  'ly': 'Voxel number in Y-direction',
                  'lz': 'Voxel number in Z-direction',
                  'divide_number': 'Upscale multiplier(1~)',
                  'mesh_size': 'Mesh size/Voxel size(0~1)',
                  'dis_y': 'Y Compression ratio(-1~1)',
                  'material_modulus': "Young's modulus(MPa)",
                  'poissons_ratio': "Poisson's ratio(0~1)",
                  'density': 'Material density(ton/mm3)',
                  'MaxRF22': 'Maximum RF22(N)',
                  'penalty_coefficient': 'Penalty coefficient',
                  'sigma': 'Sigma for filtering',
                  'threshold': 'Threshold for filtering',
                  'n_cpus': 'CPU cores for abaqus',
                  'n_gpus': 'GPU cores for abaqus',
                  'timeout': 'Timeout of validation process(s)'}
    return {v: k for k, v in dictionary.items()}.get(s) if flip else dictionary.get(s)


class Visualizer:
    def __init__(self, conn_to_gui=None):
        self.conn_to_gui = conn_to_gui
        self.ref_x = None
        self.ref_y = None
        self.all_lower_bounds = dict()
        self.all_datum_hv = dict()
        if conn_to_gui is None:
            self.figure, self.axes = plt.subplots(nrows=1, ncols=2,
                                                  figsize=((1400 - 5) / 100, (700 - 5) / 100), dpi=100)
            self.axes[0].set(title='Pareto Fronts', xlabel='Objective function 1', ylabel='Objective function 2')
            self.axes[1].set(title='Hyper Volume by Generation', xlabel='Generation', ylabel='Hyper volume')
            self.axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    def plot(self, gen_num: int, pareto_1_sorted: np.ndarray, pareto_2_sorted: np.ndarray,
             use_manual_rp: bool = False, ref_x: float = 0.0, ref_y: float = 0.0) -> None:
        if use_manual_rp:
            self.ref_x, self.ref_y = ref_x, ref_y
        elif (self.ref_x is None) or (self.ref_y is None):
            self.ref_x = pareto_1_sorted[-1]
            self.ref_y = pareto_2_sorted[0]
        else:
            if pareto_1_sorted[-1] > self.ref_x:
                self.ref_x = pareto_1_sorted[-1]
            if pareto_2_sorted[0] > self.ref_y:
                self.ref_y = pareto_2_sorted[0]
        _datum_hv = get_datum_hv(pareto_1_sorted, pareto_2_sorted)
        _lower_bounds = [pareto_1_sorted[0], pareto_2_sorted[-1]]
        self.all_datum_hv.update({gen_num: _datum_hv})
        self.all_lower_bounds.update({gen_num: _lower_bounds})
        _all_hv = {key: get_hv_from_datum_hv(self.all_datum_hv[key], self.all_lower_bounds[key],
                                             ref_x=self.ref_x, ref_y=self.ref_y) for key in self.all_datum_hv.keys()}
        _generations, _hvs = zip(*_all_hv.items())
        _file_name = '_plotting_data_'
        if os.path.isfile(_file_name):
            with open(_file_name, mode='rb') as f_read:
                _read_data = pickle.load(f_read)
            with open(_file_name, mode='wb') as f_write:
                _read_data.update({gen_num: (pareto_1_sorted, pareto_2_sorted)})
                pickle.dump(_read_data, f_write)
        else:
            with open(_file_name, mode='wb') as f_write:
                pickle.dump({gen_num: (pareto_1_sorted, pareto_2_sorted)}, f_write)
        if self.conn_to_gui is not None:
            self.conn_to_gui.send((pareto_1_sorted, pareto_2_sorted, _all_hv))
        else:
            color_1 = np.random.rand(3, )
            color_2 = np.random.rand(3, )
            plot_options = {
                'marker': np.random.choice(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')),
                'color': color_1,
                'markeredgecolor': color_1,
                'markerfacecolor': color_2,
                'markersize': 8,
                'markeredgewidth': 2}
            print(f'[VISUALIZE] Generation {gen_num}:')
            print(f'> Objective function 1:{pareto_1_sorted}')
            print(f'> Objective function 2:{pareto_2_sorted}\n')
            self.axes[0].plot(pareto_1_sorted, pareto_2_sorted, **plot_options)
            self.axes[1].clear()
            for generation, hv, line in zip(_generations, _hvs, self.axes[0].lines):
                self.axes[1].scatter(generation, hv, marker=line.get_marker(), c=[line.get_markerfacecolor()],
                                     edgecolors=line.get_markeredgecolor(), s=line.get_markersize() ** 2,
                                     linewidth=line.get_markeredgewidth())
            self.axes[0].grid(True)
            self.axes[1].grid(True)

    def visualize(self, params, w, use_manual_rp, ref_x=0.0, ref_y=0.0):
        topo_next_parent, result_next_parent = parent_import(w + 1)
        fitness_values_next_parent = evaluate_fitness_values(topo=topo_next_parent, result=result_next_parent,
                                                             params=params)
        fitness_pareto_next_parent = find_pareto_front_points(costs=fitness_values_next_parent, return_index=False)
        self.plot(gen_num=w,
                  pareto_1_sorted=fitness_pareto_next_parent[:, 0], pareto_2_sorted=fitness_pareto_next_parent[:, 1],
                  use_manual_rp=use_manual_rp, ref_x=ref_x, ref_y=ref_y)


if __name__ == '__main__':
    end_gen = 39
    set_path = r'F:\shshsh\data-23-1-4'
    parameters = Parameters()
    parameters.post_initialize()
    os.chdir(set_path)
    visualizer = Visualizer(conn_to_gui=None)

    for gen_idx in range(-1, end_gen):
        visualizer.visualize(params=parameters, w=gen_idx + 1, use_manual_rp=False)
    plt.show()
    input('Press enter to exit')
