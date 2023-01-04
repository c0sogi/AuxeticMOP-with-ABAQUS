from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import numpy as np
import random

executeOnCaeStartup()

DEFAULT_PART_NAME = 'voxel'
VOXEL_SIZE = 5.0
CELLS_PER_CUBE = 10
CUBE_SIZE = VOXEL_SIZE * CELLS_PER_CUBE
CSV_PATH = r'D:\pythoncode\22-12-28\data - original\topo_offspring_1.csv'


def model_generation(model_name):
    session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
    return mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)


def anisotropic_material_generation(model, part, voxel_size, e11, e22, e33, v12, v13, v23, g12, g13, g23):
    model.Material(name='Material-1')
    model.materials['Material-1'].Elastic(table=((e11, e22, e33, v12, v13, v23, g12, g13, g23),),
                                          type=ENGINEERING_CONSTANTS)
    model.HomogeneousSolidSection(material='Material-1', name='Section-1', thickness=None)

    mid_coordinate = voxel_size / 2
    cells = part.cells.findAt(((mid_coordinate, mid_coordinate, mid_coordinate),))
    region = regionToolset.Region(cells=cells)
    part.SectionAssignment(region=region, sectionName='Section-1',
                           offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',
                           thicknessAssignment=FROM_SECTION)
    part.MaterialOrientation(additionalRotationType=ROTATION_NONE, axis=AXIS_1,
                             fieldName='', localCsys=None, orientationType=GLOBAL,
                             region=region, stackDirection=STACK_3)


def part_generation(model, part_name, voxel_size):
    part = model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    s = model.ConstrainedSketch(name='__profile__', sheetSize=2 * voxel_size)
    s.rectangle(point1=(0.0, 0.0), point2=(voxel_size, voxel_size))
    part.BaseSolidExtrude(sketch=s, depth=voxel_size)

    f = part.faces
    mid_coordinate = voxel_size / 2
    for axis_idx, face in enumerate(('x', 'y', 'z')):
        for face_idx, face_position in enumerate((0, voxel_size)):
            find_face_at_point = [[mid_coordinate, mid_coordinate, mid_coordinate], ]
            find_face_at_point[0][axis_idx] = face_position
            part.Set(faces=f.findAt(find_face_at_point), name='voxel_{}_{}'.format(face, face_idx))
    return part


def mesh_generation(part, mesh_size):
    part.seedPart(size=mesh_size, minSizeFactor=mesh_size)
    part.generateMesh()


def assembly_generation(model, part, topo_arr, voxel_size, cells_per_cube):
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    for ix in range(cells_per_cube):
        for iy in range(cells_per_cube):
            for iz in range(cells_per_cube):
                if topo_arr[ix, iy, iz]:
                    instance_name = 'voxel-{}-{}-{}'.format(ix, iy, iz)
                    assembly.Instance(name=instance_name, part=part, dependent=ON)
                    assembly.translate(instanceList=(instance_name,),
                                       vector=(ix * voxel_size, iy * voxel_size, iz * voxel_size))
                else:
                    continue
    for axis_idx, face in enumerate(('x', 'y', 'z')):
        for face_idx, face_position in enumerate((0, voxel_size)):
            sets = []
            for axis_1 in range(cells_per_cube):
                for axis_2 in range(cells_per_cube):
                    idx_xyz = [axis_1, axis_2]
                    idx_xyz.insert(axis_idx, 0 if face_idx == 0 else cells_per_cube - 1)
                    if topo_arr[idx_xyz[0], idx_xyz[1], idx_xyz[2]]:
                        instance_name = 'voxel-{}-{}-{}'.format(idx_xyz[0], idx_xyz[1], idx_xyz[2])
                        set_name = 'voxel_{}_{}'.format(face, face_idx)
                        sets.append(assembly.allInstances[instance_name].sets[set_name])
            assembly.SetByBoolean(name='cube_{}_{}'.format(face, face_idx), sets=sets)
    return assembly


def step_generation(model, step_name, previous_step_name='Initial', maxNumInc=100000, initialInc=1e-05, minInc=1e-15):
    step = model.StaticStep(name=step_name, previous=previous_step_name,
                            maxNumInc=maxNumInc, initialInc=initialInc, minInc=minInc)
    return step


def set_displacement(model, assembly, bc_name, set_name, step_name, displacement):
    region = assembly.sets[set_name]
    model.DisplacementBC(name=bc_name, createStepName=step_name, region=region,
                         u1=displacement[0], u2=displacement[1], u3=displacement[2],
                         ur1=displacement[3], ur2=displacement[4], ur3=displacement[5],
                         amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)


def set_encastre(model, assembly, bc_name, set_name, step_name):
    region = assembly.sets[set_name]
    model.EncastreBC(name=bc_name, createStepName=step_name, region=region, localCsys=None)


def merge_instances(assembly, merged_part_name, cube_size):
    instances = assembly.instances.values()
    merged_part = assembly.InstanceFromBooleanMerge(name=merged_part_name, instances=instances, originalInstances=DELETE, mergeNodes=BOUNDARY_ONLY, nodeMergingTolerance=1e-06, domain=MESH)
    return merged_part


if __name__ == '__main__':
    topo_arr = np.genfromtxt(CSV_PATH, delimiter=',', dtype=int)[0].reshape(
        (CELLS_PER_CUBE, CELLS_PER_CUBE, CELLS_PER_CUBE))
    current_model = model_generation(model_name='Model-1')
    current_part = part_generation(model=current_model, part_name=DEFAULT_PART_NAME, voxel_size=VOXEL_SIZE)
    anisotropic_material_generation(current_model, current_part, VOXEL_SIZE,
                                    1500, 1200, 1500, 0.35, 0.35, 0.35, 450, 550, 450)
    mesh_generation(part=current_part, mesh_size=VOXEL_SIZE * 0.25)

    current_assembly = assembly_generation(model=current_model, part=current_part, topo_arr=topo_arr,
                                           voxel_size=VOXEL_SIZE, cells_per_cube=CELLS_PER_CUBE)
    # merged_part = merge_instances(assembly=current_assembly, merged_part_name='cubic', cube_size=CUBE_SIZE)
    # y_axis = current_part.DatumAxisByPrincipalAxis(principalAxis=YAXIS)
    # datum_planes = [current_part.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=i * 0.1) for i in range(CELLS_PER_CUBE)]
