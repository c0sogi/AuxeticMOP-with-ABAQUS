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


@dataclass(kw_only=True)
class Parameters:
    abaqus_script_name: str = 'ABQ.py'
    abaqus_execution_mode: str = 'script'
    mode: str = 'GA'
    evaluation_version: str = 'ver3'
    restart_pop: int = 0
    ini_pop: int = 1
    end_pop: int = 100
    ini_gen: int = 1
    end_gen: int = 50
    mutation_rate: float = 0.1
    unit_l: float = 3
    lx: int = 10
    ly: int = 10
    lz: int = 10
    divide_number: int = 1
    mesh_size: float = 0.25
    dis_y: float = -0.005
    material_modulus: float = 1100
    poissons_ratio: float = 0.4
    density: float = 1
    MaxRF22: float = 0.01
    penalty_coefficient: float = 0.1
    sigma: float = 1
    threshold: float = 0.5
    n_cpus: int = 16
    n_gpus: int = 0
    timeout: float = 0.5

    def post_initialize(self):
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


PARAMETER_FILE_NAME = '_PARAMETERS_'
padx = 5
pady = 5
left_width = 1400  # original: 400
right_width = 400
height = 750
button_width = 10
parameters = Parameters()
parameters_dict = asdict(parameters)


class App:
    def __init__(self, conn=None):
        self.conn = conn
        self.ready_to_run = False
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        self.root.title('APUCI: Abaqus-Python Unified Control Interface')
        self.root.config(background='#FFFFFF')
        self.root.resizable(False, False)

        self.up_frame = tk.Frame(self.root, width=left_width + right_width + 2 * padx, height=100)
        self.up_frame.config(background='#FFFFFF')
        self.up_frame.grid(row=0, column=0, padx=padx, pady=pady / 2)
        self.up_frame.grid_propagate(False)
        self.down_frame = tk.Frame(self.root, width=right_width + left_width + 2 * padx, height=height)
        self.down_frame.config(background='#FFFFFF')
        self.left_frame = tk.Frame(self.down_frame, width=left_width, height=height, padx=padx, pady=pady)
        self.right_frame = tk.Frame(self.down_frame, width=right_width, height=height, padx=padx, pady=pady)

        self.set_path_title = tk.Label(self.up_frame, text='Abaqus script와 Parent 파일이 들어있는 폴더를 골라 주세요')
        self.set_path_title.config(background='#FFFFFF')
        self.set_path_display = tk.Listbox(self.up_frame, width=50, height=1)
        self.set_path_btn = tk.Button(self.up_frame, text='폴더 찾기', width=8, command=self.onclick_set_path_button)
        self.submit_btn = tk.Button(self.right_frame, width=button_width, text='프리셋 저장',
                                    command=self.onclick_submit_btn)
        self.exit_btn = tk.Button(self.right_frame, width=button_width, text='종료',
                                  command=self.onclick_exit_btn, background='#FF0000', foreground='#FFFFFF')
        self.set_default_btn = tk.Button(self.right_frame, width=button_width, text='기본값 로드',
                                         command=self.onclick_set_default_btn)
        self.setPath = tk.StringVar()
        self.string_vars = [tk.StringVar() for _ in Parameters.__dict__]
        self.figure, self.ax = plt.subplots(nrows=1, ncols=2,
                                            figsize=((left_width - padx) / 100, (height - pady) / 100), dpi=100)
        self.bar = FigureCanvasTkAgg(self.figure, self.left_frame)
        self.bar.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

        self.set_path_title.pack()
        self.set_path_display.pack()
        self.set_path_btn.pack()
        self.root.mainloop()

    def callback(self):
        self.root.quit()

    def show_canvas(self):
        self.ax[0].set(title='Pareto Fronts', xlabel='Objective function 1', ylabel='Objective function 2')
        self.ax[1].set(title='Hyper Volume by Generation', xlabel='Generation', ylabel='Hyper volume')
        self.ax[0].grid(True)
        self.ax[1].grid(True)
        self.ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        if self.conn is not None:
            self.update_canvas()

    def update_canvas(self, polling_rate=1):
        if self.conn.poll():
            try:
                print('[GUI] Trying to receive plot data...')
                px, py, sx, sy = self.conn.recv()
                print('[GUI] I received something!', px, py, sx, sy)
                color_1, color_2 = np.random.rand(3, ), np.random.rand(3, )
                plot_options = {
                    'marker': np.random.choice(
                        ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')),
                    'color': color_1,
                    'markeredgecolor': color_1,
                    'markerfacecolor': color_2,
                    'markersize': 8,
                    'markeredgewidth': 2}
                self.ax[0].plot(px, py, **plot_options)
                self.ax[1].scatter(sx, sy, marker=plot_options['marker'], edgecolors=plot_options['markeredgecolor'],
                                   facecolor=plot_options['markerfacecolor'], linewidth=plot_options['markeredgewidth'],
                                   s=plot_options['markersize'] ** 2)
                self.bar.draw()
            except Exception as error_message:
                print('[GUI] An plotting error occurred:', error_message)
        self.root.after(int(1000/polling_rate), self.update_canvas)

    def onclick_set_path_button(self):
        try:
            folder_path = askdirectory(initialdir="./")
            if folder_path:
                self.set_path_display.delete(0, "end")
                self.set_path_display.insert(0, folder_path)
            self.setPath.set(self.set_path_display.get(0))
            os.chdir(self.setPath.get())

            params_main_already_exists = True if os.path.isfile(PARAMETER_FILE_NAME) else False
            if params_main_already_exists:
                with open(PARAMETER_FILE_NAME, mode='rb') as f:
                    self.set_path_display.config(background='#00FF00')
                    self.set_path_title.config(text='미리 설정된 파라미터 값을 찾았습니다.')
                    self.show_parameters(loaded=pickle.load(f))
                    self.ready_to_run = False
            else:
                self.set_path_display.config(background='#FF0000')
                self.set_path_title.config(text='파라미터를 찾지 못했습니다. 밑에서 설정해주세요.')
                self.show_parameters(loaded=False)
                self.ready_to_run = False

        except Exception as error_message:
            messagebox.showerror("Error", f"오류가 발생했습니다:\n{error_message}")

    def onclick_set_default_btn(self):
        for idx, value in enumerate(parameters_dict.values()):
            self.string_vars[idx].set(value)

    def return_radiobutton_frame_instead_of_entry(self, key, i, d):  # right_Frame
        radiobutton_frame = tk.Frame(self.right_frame, width=left_width - 2 * padx,
                                     height=height / len(parameters_dict) - pady)
        if len(d[key]) == 2:
            lb_width = 25
            rb_width = 9
        elif len(d[key]) == 3:
            lb_width = 24
            rb_width = 5
        else:
            raise ValueError('체크박스의 개수가 3개 이하가 되도록 해주세요.')
        lb = tk.Label(radiobutton_frame, width=lb_width, text=translator(s=key, to_korean=True), anchor='w')
        lb.grid(row=0, column=0)

        for menu_idx, menu in enumerate(d[key]):
            rb = tk.Radiobutton(radiobutton_frame, text=menu, variable=self.string_vars[i],
                                width=rb_width, value=menu, tristatevalue=menu)
            rb.grid(row=0, column=menu_idx + 1)
        return radiobutton_frame

    def onclick_submit_btn(self):
        for params_idx, key in enumerate(parameters_dict.keys()):
            parameters_dict[key] = string_to_int_or_float_or_string(self.string_vars[params_idx].get())
        if self.ready_to_run and self.conn is not None:
            self.conn.send((self.setPath.get(), Parameters(**parameters_dict)))
            self.submit_btn.config(background='#0000FF', foreground='#FFFFFF', text='실행 중')
        else:
            with open(PARAMETER_FILE_NAME, mode='wb') as f:
                pickle.dump(parameters_dict, f)
                print(f'[GUI] Dumping to "{os.getcwd()}" Complete')
                for key, value in parameters_dict.items():
                    print(f'- {key}: {value}')
            self.submit_btn.config(background='#00FF00', text='실행')
            self.ready_to_run = True

    def onclick_exit_btn(self):
        print('[GUI] Closing GUI...')
        self.root.quit()

    def show_parameters(self, loaded):
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
        for i, key in enumerate(asdict(Parameters()).keys()):
            if key in radiobutton_name_dict.keys():
                rbf = self.return_radiobutton_frame_instead_of_entry(key=key, i=i, d=radiobutton_name_dict)
                rbf.grid(row=i, column=0)

            else:
                ef = tk.Frame(self.right_frame, width=left_width - 2 * padx,
                              height=height / len(asdict(Parameters()).keys()) - pady)
                lb = tk.Label(ef, width=25, text=translator(s=key, to_korean=True), anchor='w')
                ee = tk.Entry(ef, width=25, textvariable=self.string_vars[i])
                ef.grid(row=i, column=0)
                lb.grid(row=0, column=0)
                ee.grid(row=0, column=1)
            self.string_vars[i].set('')

        empty_label = tk.Label(self.right_frame)
        empty_label.grid(row=len(asdict(Parameters()).keys()))
        self.submit_btn.grid(row=len(asdict(Parameters()).keys()) + 1)
        self.exit_btn.grid(row=len(asdict(Parameters()).keys()) + 2)
        self.set_default_btn.grid(row=len(asdict(Parameters()).keys()) + 3)

        if loaded:
            for params_idx, (key, value) in enumerate(loaded.items()):
                self.string_vars[params_idx].set(value)
        self.show_canvas()


def string_to_int_or_float_or_string(s):
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


def translator(s, to_korean):
    dictionary = {'abaqus_script_name': 'ABAQUS 스크립트 파일명',
                  'abaqus_execution_mode': 'ABAQUS 실행 모드',
                  'mode': '모드',
                  'evaluation_version': '평가 버전',
                  'restart_pop': '재시작 Population',
                  'ini_pop': '첫 Population',
                  'end_pop': '끝 Population',
                  'ini_gen': '첫 Generation',
                  'end_gen': '끝 Generation',
                  'mutation_rate': '돌연변이율(0~1)',
                  'unit_l': 'Voxel 크기(mm)',
                  'lx': 'X방향 Voxel 수',
                  'ly': 'Y방향 Voxel 수',
                  'lz': 'Z방향 Voxel 수',
                  'divide_number': '정밀도(1이상 자연수)',
                  'mesh_size': '메쉬 정밀도(0~1)',
                  'dis_y': 'Y방향 압축률(-1~1)',
                  'material_modulus': '재료 영률값(MPa)',
                  'poissons_ratio': '포아송 비(0~1)',
                  'density': '재료 밀도(ton/mm3)',
                  'MaxRF22': 'RF22 최댓값(N)',
                  'penalty_coefficient': '패널티 계수',
                  'sigma': '시그마',
                  'threshold': '임계값',
                  'n_cpus': 'CPU 코어 수',
                  'n_gpus': 'GPU 코어 수',
                  # 'add_probability': '트리 덧셈 확률(0~1)',
                  'timeout': '타임 아웃(s)'}
    return dictionary.get(s) if to_korean else {v: k for k, v in dictionary.items()}.get(s)


def plot_test(ax, px, py, sx, sy):
    color_1 = np.random.rand(3, )
    color_2 = np.random.rand(3, )
    plot_options = {
        'marker': np.random.choice(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')),
        'color': color_1,
        'markeredgecolor': color_1,
        'markerfacecolor': color_2,
        'markersize': 8,
        'markeredgewidth': 2,

    }
    ax[0].plot(px, py, **plot_options)
    ax[1].scatter(sx, sy, marker=plot_options['marker'], edgecolors=plot_options['markeredgecolor'],
                  facecolor=plot_options['markerfacecolor'], linewidth=plot_options['markeredgewidth'],
                  s=plot_options['markersize'] ** 2)


if __name__ == '__main__':
    App()
