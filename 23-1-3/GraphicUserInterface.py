import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter import messagebox
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator

padx = 5
pady = 5
left_width = 1400  # original: 400
right_width = 400
height = 700
button_width = 8

params_main = {
    'abaqus_script_name': 'ABQ.py',
    'abaqus_execution_mode': 'noGUI',
    'mode': 'GA',
    'evaluation_version': 'ver3',
    'restart_pop': 0,
    'ini_pop': 1,
    'end_pop': 100,
    'ini_gen': 1,
    'end_gen': 50,
    'mutation_rate': 0.05,
    'unit_l': 3,
    'lx': 10,
    'ly': 10,
    'lz': 10,
    'divide_number': 1,
    'mesh_size': 0.25,
    'dis_y': -0.005,
    'material_modulus': 1100,
    'poissons_ratio': 0.4,
    'density': 1,
    'MaxRF22': 0.01,
    'penalty_coefficient': 0.1,
    'sigma': 1,
    'threshold': 0.5,
    'n_cpus': 8,
    'n_gpus': 0
}
params_main_default = params_main.copy()
params_main_loaded = [dict()]


class App:
    def __init__(self, conn):
        self.conn = conn
        self.ready_to_run = False
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        self.root.title('MPCI: Main Parameter Configuration Interface')
        self.root.config(background='#FFFFFF')
        self.root.resizable(False, False)
        self.up_frame = tk.Frame(self.root, width=left_width + right_width + 2 * padx, height=100)
        self.up_frame.config(background='#FFFFFF')
        self.set_path_title = tk.Label(self.up_frame, text='Abaqus script와 Parent 파일이 들어있는 폴더를 골라 주세요')
        self.set_path_title.config(background='#FFFFFF')
        self.set_path_display = tk.Listbox(self.up_frame, width=50, height=1)
        self.set_path_btn = tk.Button(self.up_frame, text='폴더 찾기', width=8, command=self.onclick_set_path_button)
        self.up_frame.grid(row=0, column=0, padx=padx, pady=pady / 2)
        self.up_frame.grid_propagate(False)
        self.set_path_title.pack()
        self.set_path_display.pack()
        self.set_path_btn.pack()
        self.setPath = tk.StringVar()
        self.string_vars = [tk.StringVar() for _ in params_main.keys()]
        self.root.mainloop()

    def callback(self):
        self.root.quit()

    def show_canvas(self):
        self.figure, self.ax = plt.subplots(nrows=1, ncols=2,
                                            figsize=((left_width - padx) / 100, (height - pady) / 100), dpi=100)
        self.bar = FigureCanvasTkAgg(self.figure, self.left_frame)
        self.bar.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        self.ax[0].set(title='Objective Function: 1, 2', xlabel='Objective function 1', ylabel='Objective function 2')
        self.ax[1].set(title='Hypervolume by generation', xlabel='Iteration', ylabel='Hypervolume')
        self.ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        self.update_canvas()

    def scatter(self, axis_idx, i, j):
        print('Scattering: ', i, j)
        self.ax[axis_idx].scatter(i, j)
        self.bar.draw()

    def plot(self, axis_idx, i, j):
        print('Plotting: ', i, j)
        self.ax[axis_idx].plot(i, j, marker='o', color='#2ca02c')
        self.bar.draw()

    def update_canvas(self):
        if self.conn.poll():
            try:
                print('Trying to receive plot data...')
                px, py, sx, sy = self.conn.recv()
                print('I received something!', px, py, sx, sy)
                self.ax[0].plot(px, py)
                self.ax[1].scatter(sx, sy)
                self.bar.draw()
            except:
                print('An plotting error occured!')
        self.root.after(1000, self.update_canvas)

    def onclick_set_path_button(self):
        try:
            folder_path = askdirectory(initialdir="./")
            if folder_path:
                self.set_path_display.delete(0, "end")
                self.set_path_display.insert(0, folder_path)
            self.setPath.set(self.set_path_display.get(0))
            os.chdir(self.setPath.get())

            params_main_already_exists = True if os.path.isfile('./PARAMS_MAIN') else False
            if params_main_already_exists:
                with open('./PARAMS_MAIN', mode='rb') as f:
                    params_main_loaded[0] = pickle.load(f)
                    self.set_path_display.config(background='#00FF00')
                    self.set_path_title.config(text='미리 설정된 파라미터 값을 찾았습니다.')
                    self.show_parameters(loaded=True)
                    self.ready_to_run = False
            else:
                self.set_path_display.config(background='#FF0000')
                self.set_path_title.config(text='파라미터를 찾지 못했습니다. 밑에서 설정해주세요.')
                self.show_parameters(loaded=False)
                self.ready_to_run = False

        except ValueError as e:
            messagebox.showerror("Error", f"오류가 발생했습니다:\n{e}")
        except:
            messagebox.showerror("Error", "오류가 발생했습니다")

    def onclick_set_default_btn(self):
        for idx, key in enumerate(params_main_default.keys()):
            self.string_vars[idx].set(params_main_default[key])

    def string_to_int_or_float_or_string(self, s):
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

    def return_radiobutton_frame_instead_of_entry(self, key, i, d):  # right_Frame
        radiobutton_frame = tk.Frame(self.right_frame, width=left_width - 2 * padx,
                                     height=height / len(params_main.keys()) - pady)
        if len(d[key]) == 2:
            lb_width = 25
            rb_width = 9
        elif len(d[key]) == 3:
            lb_width = 24
            rb_width = 5
        else:
            raise ValueError('체크박스의 개수가 3개 이하가 되도록 해주세요.')
        lb = tk.Label(radiobutton_frame, width=lb_width, text=key, anchor='w')
        lb.grid(row=0, column=0)

        for menu_idx, menu in enumerate(d[key]):
            rb = tk.Radiobutton(radiobutton_frame, text=menu, variable=self.string_vars[i],
                                width=rb_width, value=menu, tristatevalue=menu)
            rb.grid(row=0, column=menu_idx + 1)
        return radiobutton_frame

    def onclick_submit_btn(self):
        for params_idx, key in enumerate(params_main.keys()):
            params_main[key] = self.string_to_int_or_float_or_string(self.string_vars[params_idx].get())
        if self.ready_to_run:
            self.conn.send(self.setPath.get())
        else:
            with open('./PARAMS_MAIN', mode='wb') as f:
                pickle.dump(params_main, f)
                print('Dumping complete:\n', params_main)
            self.submit_btn.config(background='#00FF00', text='실행')
            self.ready_to_run = True

    def onclick_exit_btn(self):
        print('Closing GUI...')
        self.root.quit()

    def show_parameters(self, loaded):
        radiobutton_name_dict = {
            'abaqus_execution_mode': ('noGUI', 'script'),
            'mode': ('GA', 'random'),
            'evaluation_version': ('ver1', 'ver2', 'ver3')
        }
        self.down_frame = tk.Frame(self.root, width=right_width + left_width + 2 * padx, height=height)
        self.down_frame.config(background='#FFFFFF')
        self.left_frame = tk.Frame(self.down_frame, width=left_width, height=height, padx=padx, pady=pady)
        # plot_frame[0] = left_frame  # $
        self.right_frame = tk.Frame(self.down_frame, width=right_width, height=height, padx=padx, pady=pady)
        self.down_frame.grid(row=1, column=0, padx=padx, pady=pady / 2)
        self.down_frame.grid_propagate(False)
        self.left_frame.grid(row=1, column=0, padx=padx, pady=pady / 2)
        self.left_frame.grid_propagate(False)
        self.right_frame.grid(row=1, column=1, padx=padx, pady=pady / 2)
        self.right_frame.grid_propagate(False)

        for i, key in enumerate(params_main.keys()):
            if key in radiobutton_name_dict.keys():
                rbf = self.return_radiobutton_frame_instead_of_entry(key=key, i=i, d=radiobutton_name_dict)
                rbf.grid(row=i, column=0)

            else:
                ef = tk.Frame(self.right_frame, width=left_width - 2 * padx,
                              height=height / len(params_main.keys()) - pady)
                lb = tk.Label(ef, width=25, text=key, anchor='w')
                ee = tk.Entry(ef, width=25, textvariable=self.string_vars[i])
                ef.grid(row=i, column=0)
                lb.grid(row=0, column=0)
                ee.grid(row=0, column=1)
            self.string_vars[i].set('')

        empty_label = tk.Label(self.right_frame)
        self.submit_btn = tk.Button(self.right_frame, width=button_width, text='저장', command=self.onclick_submit_btn)
        self.exit_btn = tk.Button(self.right_frame, width=button_width, text='종료',
                                  command=self.onclick_exit_btn, background='#FF0000')
        self.set_default_btn = tk.Button(self.right_frame, width=button_width, text='추천값 로드',
                                         command=self.onclick_set_default_btn)

        empty_label.grid(row=len(params_main.keys()))
        self.submit_btn.grid(row=len(params_main.keys()) + 1)
        self.exit_btn.grid(row=len(params_main.keys()) + 2)
        self.set_default_btn.grid(row=len(params_main.keys()) + 3)

        if loaded:
            for params_idx, key in enumerate(params_main.keys()):
                params_main[key] = params_main_loaded[0][key]
                self.string_vars[params_idx].set(params_main[key])
        self.show_canvas()