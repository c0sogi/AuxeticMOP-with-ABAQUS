from dataclasses import dataclass


@dataclass  # Use @dataclass(kw_only=True) for Python version >= 3.10
class Parameters:
    abaqus_script: str = 'ABQ.py'  # abaqus python script filename e.g., ABQ.py
    abaqus_mode: str = 'script'  # noGUI: without abaqus gui, script: with abaqus gui
    mode: str = 'GA'  # GA mode
    evaluation_version: str = 'ver3'  # fitness value evaluation mode
    restart_pop: int = 0  # 0 for no-restart, 1~ for initial restarting population
    ini_pop: int = 1  # First population number, default: 1
    end_pop: int = 100  # Last population number
    ini_gen: int = 1  # First generation number, default: 1
    end_gen: int = 50  # Last generation number
    mutation_rate: float = 0.1  # mutation process option
    unit_l: float = 3  # Voxel length
    lx: int = 5  # Number of voxels in x-direction
    ly: int = 5  # Number of voxels in y-direction
    lz: int = 5  # Number of voxels in z-direction
    divide_number: int = 1  # up-scaling factor
    mesh_size: float = 0.5  # abaqus meshing option
    dis_y: float = -0.005  # abaqus boundary condition option
    material_modulus: float = 1100  # abaqus material property option
    poisson_ratio: float = 0.4  # abaqus material property option
    density: float = 1  # abaqus material property option
    MaxRF22: float = 0.01  # fitness value evaluation option
    penalty_coefficient: float = 0.1  # fitness value evaluation option
    sigma: float = 1  # filtering option
    threshold: float = 0.5  # filtering option
    n_cpus: int = 1  # abaqus option
    n_gpus: int = 0  # abaqus option
    timeout: float = 0.5  # validation process option

    def post_initialize(self):  # call this method to set initial values to real value to be used
        self.lx *= self.divide_number
        self.ly *= self.divide_number
        self.lz *= self.divide_number  # number of voxels after increasing resolution
        self.unit_l /= self.divide_number
        unit_lx_total = self.lx * self.unit_l
        unit_ly_total = self.ly * self.unit_l
        unit_lz_total = self.lz * self.unit_l
        self.mesh_size *= self.unit_l
        self.dis_y *= unit_ly_total  # boundary condition (displacement)
        self.MaxRF22 *= unit_lx_total * unit_lz_total * self.material_modulus
