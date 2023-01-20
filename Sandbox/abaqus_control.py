from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import numpy as np

executeOnCaeStartup()

DEFAULT_PART_NAME = 'voxel'
CUBE_NAME = 'cubc'
CUBE_333_NAME = 'cubic_333'
VOXEL_SIZE = 3.0
CELLS_PER_CUBE = 10
MESH_RATIO = 0.5
CUBE_333_SIZE = VOXEL_SIZE * 2 * 3 * CELLS_PER_CUBE


def model_generation(model_name):
    session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
    return mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)


def anisotropic_material_generation(model, part, density, e11, e22, e33, v12, v13, v23, g12, g13, g23):
    model.Material(name='Material-1')
    model.materials['Material-1'].Density(table=((density,),))
    model.materials['Material-1'].Elastic(table=((e11, e22, e33, v12, v13, v23, g12, g13, g23),),
                                          type=ENGINEERING_CONSTANTS)
    model.HomogeneousSolidSection(material='Material-1', name='Section-1', thickness=None)
    elements = part.elements.getByBoundingBox(0, 0, 0, CUBE_333_SIZE, CUBE_333_SIZE, CUBE_333_SIZE)
    region = regionToolset.Region(elements=elements)
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


def merge_instances(assembly, merged_part_name):
    instances = assembly.instances.values()
    merged = assembly.InstanceFromBooleanMerge(name=merged_part_name, instances=instances,
                                               originalInstances=DELETE, mergeNodes=BOUNDARY_ONLY,
                                               nodeMergingTolerance=1e-06, domain=MESH)
    del assembly.features[merged_part_name + '-1']
    return merged


def make_333(merged_part_name):
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                instance_name = '{}-{}-{}-{}'.format(CUBE_NAME, ix, iy, iz)
                spacing = CELLS_PER_CUBE * 2 * VOXEL_SIZE
                ins = current_assembly.Instance(name=instance_name, part=current_model.parts[CUBE_NAME], dependent=ON)
                ins.translate(vector=(ix * spacing, iy * spacing, iz * spacing))
    current_assembly.InstanceFromBooleanMerge(name=merged_part_name, instances=current_assembly.instances.values(),
                                              originalInstances=DELETE, mergeNodes=BOUNDARY_ONLY,
                                              nodeMergingTolerance=1e-06, domain=MESH)


def quaver_to_full(quaver):
    quarter = np.concatenate((np.flip(quaver, axis=0), quaver), axis=0)
    half = np.concatenate((np.flip(quarter, axis=1), quarter), axis=1)
    full = np.concatenate((np.flip(half, axis=2), half), axis=2)
    return np.swapaxes(full, axis1=0, axis2=2)


if __name__ == '__main__':
    path = r'F:\shshsh\data-23-1-4\topo_parent_51.csv'
    pareto_model_indices = [0, ]
    all_topos = np.swapaxes(np.genfromtxt(path, delimiter=',', dtype=int).reshape((100, 10, 10, 10)), axis1=1, axis2=3)
    pareto_topos = all_topos[pareto_model_indices]

    for model_idx, pareto_topo in enumerate(pareto_topos):
        current_model = model_generation(model_name='Model-' + str(model_idx + 1))
        current_part = part_generation(model=current_model, part_name=DEFAULT_PART_NAME, voxel_size=VOXEL_SIZE)
        mesh_generation(part=current_part, mesh_size=VOXEL_SIZE * MESH_RATIO)
        current_assembly = assembly_generation(model=current_model, part=current_part,
                                               topo_arr=quaver_to_full(pareto_topo),
                                               voxel_size=VOXEL_SIZE, cells_per_cube=2 * CELLS_PER_CUBE)
        merge_instances(assembly=current_assembly, merged_part_name=CUBE_NAME)
        make_333(merged_part_name=CUBE_333_NAME)
        part_333 = current_model.parts[CUBE_333_NAME]
        anisotropic_material_generation(current_model, part_333, 1.2e-09, 1500, 1200, 1500, 0.35, 0.35, 0.35, 450, 550, 450)
        bottom_set = part_333.Set(nodes=part_333.nodes.getByBoundingBox(0, 0, 0, CUBE_333_SIZE, 0, CUBE_333_SIZE), name='bottom')
        set_encastre(model=current_model, assembly=current_assembly, bc_name='fix_bottom', set_name=CUBE_333_NAME+'-1.bottom', step_name='Initial')
        current_model.FrequencyStep(name='Modal', previous='Initial', limitSavedEigenvectorRegion=None, numEigen=12)
        mdb.Job(name='Job-{}'.format(model_idx+1), model='Model-{}'.format(model_idx+1), description='', type=ANALYSIS,
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
                scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1,
                multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
        mdb.jobs['Job-{}'.format(model_idx+1)].submit(consistencyChecking=OFF)