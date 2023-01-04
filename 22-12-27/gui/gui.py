import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter import messagebox
import os
import pickle
from time import sleep

padx = 5
pady = 5
left_width = 400
right_width = 400
height = 700
button_width = 8

set_path = ['']
params_main = {
    'abaqus_script_name': 'abaqus_script_new.py',
    'abaqus_execution_mode': 'noGUI',
    'mode': 'GA',
    'evaluation_version': 'ver3',
    'restart_pop': 0,
    'ini_pop': 1,
    'end_pop': 100,
    'ini_gen': 1,
    'end_gen': 50,
    'mutation_rate': 10,
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


def onclick_set_default_btn(string_vars):
    for idx, key in enumerate(params_main_default.keys()):
        string_vars[idx].set(params_main_default[key])


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


def return_radiobutton_frame_instead_of_entry(frame, string_vars, key, i, d):  # right_Frame
    radiobutton_frame = tk.Frame(frame, width=left_width - 2 * padx, height=height / len(params_main.keys()) - pady)
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
        rb = tk.Radiobutton(radiobutton_frame, text=menu, variable=string_vars[i],
                            width=rb_width, value=menu, tristatevalue=menu)
        rb.grid(row=0, column=menu_idx + 1)
    return radiobutton_frame


def onclick_submit_btn(string_var_list, button):
    for params_idx, key in enumerate(params_main.keys()):
        params_main[key] = string_to_int_or_float_or_string(string_var_list[params_idx].get())
    button.config(background='#00FF00', text='저장 완료')
    with open('./PARAMS_MAIN', mode='wb') as f:
        dump_dict = {**{'setPath': set_path[0]}, **params_main}
        pickle.dump(dump_dict, f)
        print('Dumping complete:\n', dump_dict)


def show_parameters(window, loaded, string_vars):
    radiobutton_name_dict = {
        'abaqus_execution_mode': ('noGUI', 'script'),
        'mode': ('GA', 'random'),
        'evaluation_version': ('ver1', 'ver2', 'ver3')
    }
    down_frame = tk.Frame(window, width=right_width + left_width + 2 * padx, height=height)
    down_frame.config(background='#FFFFFF')
    left_frame = tk.Frame(down_frame, width=left_width, height=height, padx=padx, pady=pady)
    right_frame = tk.Frame(down_frame, width=right_width, height=height, padx=padx, pady=pady)
    down_frame.grid(row=1, column=0, padx=padx, pady=pady / 2)
    down_frame.grid_propagate(False)
    left_frame.grid(row=1, column=0, padx=padx, pady=pady / 2)
    left_frame.grid_propagate(False)
    right_frame.grid(row=1, column=1, padx=padx, pady=pady / 2)
    right_frame.grid_propagate(False)

    for i, key in enumerate(params_main.keys()):
        if key in radiobutton_name_dict.keys():
            rbf = return_radiobutton_frame_instead_of_entry(frame=right_frame, string_vars=string_vars,
                                                            key=key, i=i, d=radiobutton_name_dict)
            rbf.grid(row=i, column=0)

        else:
            ef = tk.Frame(right_frame, width=left_width - 2 * padx, height=height / len(params_main.keys()) - pady)
            lb = tk.Label(ef, width=25, text=key, anchor='w')
            ee = tk.Entry(ef, width=25, textvariable=string_vars[i])
            ef.grid(row=i, column=0)
            lb.grid(row=0, column=0)
            ee.grid(row=0, column=1)
        string_vars[i].set('')



    empty_label = tk.Label(right_frame)
    submit_btn = tk.Button(right_frame, width=button_width, text='저장', command=lambda: onclick_submit_btn(
        string_var_list=string_vars,
        button=submit_btn))
    exit_btn = tk.Button(right_frame, width=button_width, text='종료',
                         command=lambda: print([sv.get() for sv in string_vars]))
    tk.Button()
    set_default_btn = tk.Button(right_frame, width=button_width, text='추천값 로드',
                                command=lambda: onclick_set_default_btn(string_vars=string_vars))

    empty_label.grid(row=len(params_main.keys()))
    submit_btn.grid(row=len(params_main.keys()) + 1)
    exit_btn.grid(row=len(params_main.keys()) + 2)
    set_default_btn.grid(row=len(params_main.keys())+3)

    if loaded:
        for params_idx, key in enumerate(params_main.keys()):
            params_main[key] = params_main_loaded[0][key]
            string_vars[params_idx].set(params_main[key])
        # submit_btn.config(background='#00FF00')


def onclick_set_path_button(window, listbox, title, string_vars):
    try:
        folder_path = askdirectory(initialdir="./")
        if folder_path:
            listbox.delete(0, "end")
            listbox.insert(0, folder_path)
        set_path[0] = listbox.get(0)
        os.chdir(set_path[0])

        params_main_already_exists = True if os.path.isfile('./PARAMS_MAIN') else False
        if params_main_already_exists:
            with open('./PARAMS_MAIN', mode='rb') as f:
                params_main_loaded[0] = pickle.load(f)
                listbox.config(background='#00FF00')
                title.config(text='미리 설정된 파라미터 값을 찾았습니다.')
                show_parameters(window=window, loaded=True, string_vars=string_vars)
        else:
            listbox.config(background='#FF0000')
            title.config(text='파라미터를 찾지 못했습니다. 밑에서 설정해주세요.')
            show_parameters(window=window, loaded=False, string_vars=string_vars)

    except ValueError as e:
        messagebox.showerror("Error", f"오류가 발생했습니다:\n{e}")
    except:
        messagebox.showerror("Error", "오류가 발생했습니다")


# PARAMS_MAIN 을 GA 에서 쓸 수 있도록 저장 해준다
def get_params_main():
    # 창을 열고 경로를 물어 본다
    root = tk.Tk()
    root.title('MPCI: Main Parameter Configuration Interface')
    root.config(background='#FFFFFF')
    root.resizable(False, False)
    string_vars = [tk.StringVar() for _ in params_main.keys()]

    up_frame = tk.Frame(root, width=left_width + right_width + 2 * padx, height=100)
    up_frame.config(background='#FFFFFF')
    set_path_title = tk.Label(up_frame, text='Abaqus script와 Parent 파일이 들어있는 폴더를 골라 주세요')
    set_path_title.config(background='#FFFFFF')
    set_path_display = tk.Listbox(up_frame, width=50, height=1)
    set_path_btn = tk.Button(up_frame, text='폴더 찾기', width=8,
                             command=lambda: onclick_set_path_button(window=root,
                                                                     listbox=set_path_display,
                                                                     title=set_path_title,
                                                                     string_vars=string_vars))

    up_frame.grid(row=0, column=0, padx=padx, pady=pady / 2)
    up_frame.grid_propagate(False)

    set_path_title.pack()
    set_path_display.pack()
    set_path_btn.pack()
    root.mainloop()

    return set_path[0]


if __name__ == '__main__':
    try:
        get_params_main()
    except Exception as e:
        print(e)
