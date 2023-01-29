from dataclasses import dataclass
import numpy as np
from typing import Union


@dataclass  # Use @dataclass(kw_only=True) for Python version >= 3.10
class Parameters:
    abaqus_mode: str = 'script'  # noGUI: without abaqus gui, script: with abaqus gui
    # mode: str = 'GA'  # GA mode
    evaluation_version: str = 'ver5'  # fitness value evaluation mode
    end_pop: int = 10  # Last population number
    end_gen: int = 50  # Last generation number
    mutation_rate: float = 0.1  # mutation process option
    unit_l: float = 3  # Voxel length
    lx: int = 5  # Number of voxels in x-direction
    ly: int = 5  # Number of voxels in y-direction
    lz: int = 5  # Number of voxels in z-direction
    divide_number: int = 1  # up-scaling factor
    mesh_size: float = 1.0  # abaqus meshing option
    dis_y: float = -0.005  # abaqus boundary condition option
    material_modulus: float = 1100  # abaqus material property option
    # poisson_ratio: float = 0.4  # abaqus material property option
    # density: float = 1  # abaqus material property option
    penalty_coefficient: float = 0.1  # fitness value evaluation option
    # sigma: float = 1  # filtering option
    # threshold: float = 0.5  # filtering option
    n_cpus: int = 1  # abaqus option
    n_gpus: int = 0  # abaqus option

    def post_initialize(self):  # call this method to set initial values to real value to be used
        self.lx *= self.divide_number
        self.ly *= self.divide_number
        self.lz *= self.divide_number  # number of voxels after increasing resolution
        self.unit_l /= self.divide_number
        self.mesh_size *= self.unit_l
        self.dis_y *= self.ly * self.unit_l  # boundary condition (displacement)


@dataclass
class JsonFormat:
    start_topology_from: int
    topologies_file_name: str
    topologies_key: str
    exit_abaqus: bool


@dataclass
class GuiParameters:
    parameter_file_name: str = '_PARAMETERS_'
    padx: int = 5  # Padding width
    pady: int = 5  # Padding height
    left_width: int = 1400  # default width: 400
    right_width: int = 400
    height: int = 750
    button_width: int = 15
    polling_rate: float = 10.0
    title: str = 'Abaqus-Python Unified Control Interface'


radiobutton_name_dict = {
    'abaqus_mode': ('noGUI', 'script'),
    'mode': ('GA', 'random'),
    'evaluation_version': ('ver3', 'ver4', 'ver5')
}


@dataclass
class FitnessDefinitions:
    vars_definitions: dict
    fitness_value_definitions: Union[tuple, list]


material_property_definitions = {
    'material_name': 'resin',
    'density': 1.2e-09,
    'engineering_constants': (1500, 1200, 1500, 0.35, 0.35, 0.35, 450, 550, 450)
}

exported_field_outputs_format = {
    'displacement': {'xMax': np.ndarray, 'yMax': np.ndarray, 'zMax': np.ndarray},
    'rotation': np.ndarray,
    'reaction_force': np.ndarray,
    'mises_stress': {'max': float, 'min': float, 'average': float}
}

fitness_definitions = {
    'ver1': FitnessDefinitions(
        vars_definitions={
            'dis11': ('displacement', 'xMax', 0),
            'dis22': ('displacement', 'yMax', 1),
            'rf22': ('reaction_force', 2),
            'max_rf22': '$max_rf22',
            'k': '@penalty_coefficient',  # The prefix @ means this is variable is from Parameters
            'total_voxels': '$total_voxels',  # The prefix $ means this is predefined variable
            'lx': '@lx',
            'ly': '@ly',
            'lz': '@lz'
        },
        fitness_value_definitions=(
            '(rf22 / max_rf22) + k * total_voxels / (lx * ly * lz)',
            '- (dis11 / dis22) + k * total_voxels / (lx * ly * lz)'
        )),
    'ver2': FitnessDefinitions(
        vars_definitions={
            'rf22': ('reaction_force', 2),
            'max_rf22': '$max_rf22',
            'total_voxels': '$total_voxels',
            'lx': '@lx',
            'ly': '@ly',
            'lz': '@lz'
        },
        fitness_value_definitions=(
            'total_voxels / (lx * ly * lz)',
            'rf22 / max_rf22'
        )),
    'ver3': FitnessDefinitions(
        vars_definitions={
            'dis11': ('displacement', 'xMax', 0),
            'dis22': ('displacement', 'yMax', 1),
            'dis33': ('displacement', 'zMax', 2),
            'k': '@penalty_coefficient',
            'total_voxels': '$total_voxels',
            'lx': '@lx',
            'ly': '@ly',
            'lz': '@lz'
        },
        fitness_value_definitions=(
            '- (dis11 / dis22) + k * (total_voxels) / (lx * ly * lz)',
            '- (dis33 / dis22) + k * (total_voxels) / (lx * ly * lz)'
        )),
    'ver4': FitnessDefinitions(
        vars_definitions={
            'max_mises': ('mises_stress', 'max'),
            'total_voxels': '$total_voxels',
            'lx': '@lx',
            'ly': '@ly',
            'lz': '@lz'
        },
        fitness_value_definitions=(
            'max_mises',
            'total_voxels / (lx * ly * lz)'
        )),
    'ver5': FitnessDefinitions(
        vars_definitions={
            'max_mises': ('mises_stress', 'max'),
            'dis11': ('displacement', 'xMax', 0),
            'dis22': ('displacement', 'yMax', 1),
            'dis33': ('displacement', 'zMax', 2)
        },
        fitness_value_definitions=(
            'max_mises',
            'max(-dis11/dis22, -dis33/dis22)'
        ))
}

translate_dictionary = {'abaqus_script': 'Filename of ABAQUS script',
                        'abaqus_mode': 'ABAQUS execution mode',
                        'mode': 'GA Mode',
                        'evaluation_version': 'GA evaluation version',
                        'restart_pop': '[P] Restart from population',
                        'ini_pop': '[P] First Population',
                        'end_pop': '[P] Populations per generation',
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
                        'poisson_ratio': "Poisson's ratio(0~1)",
                        'density': 'Material density(ton/mm3)',
                        'MaxRF22': 'Maximum RF22(N)',
                        'penalty_coefficient': 'Penalty coefficient',
                        'sigma': 'Sigma for filtering',
                        'threshold': 'Threshold for filtering',
                        'n_cpus': 'CPU cores for abaqus',
                        'n_gpus': 'GPU cores for abaqus',
                        'timeout': 'Timeout of validation process(s)'}
