import regionToolset
from abaqus import *
from abaqusConstants import *
from driverUtils import executeOnCaeStartup
import numpy as np

executeOnCaeStartup()


class MyModel:
    def __init__(self, model_name, params):
        session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
        self.model = mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)
        self.root_assembly = self.model.rootAssembly
        self.root_assembly.DatumCsysByDefault(CARTESIAN)
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(mdb.models.keys()) == 1:
            mdb.Model(name='empty_model', modelType=STANDARD_EXPLICIT)
        # del mdb.models[self.model.name]
        del self

    def create_voxel_part(self, voxel_name):
        _voxel_part = self.model.Part(name=voxel_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        _s = self.model.ConstrainedSketch(name='__profile__', sheetSize=2 * self.params['unit_l'])
        _s.rectangle(point1=(0.0, 0.0), point2=(self.params['unit_l'], self.params['unit_l']))
        _voxel_part.BaseSolidExtrude(sketch=_s, depth=self.params['unit_l'])

    def create_cube_part(self, voxel_name, cube_name, topo_arr):
        for ix in range(topo_arr.shape[0]):
            for iy in range(topo_arr.shape[1]):
                for iz in range(topo_arr.shape[2]):
                    if topo_arr[ix, iy, iz]:
                        _instance_name = '{}-{}-{}-{}'.format(cube_name, ix, iy, iz)
                        self.root_assembly.Instance(name=_instance_name, part=self.model.parts[voxel_name],
                                                    dependent=ON)
                        self.root_assembly.translate(instanceList=(_instance_name,),
                                                     vector=(ix * self.params['unit_l'],
                                                             iy * self.params['unit_l'],
                                                             iz * self.params['unit_l']))
        self.merge_mesh_of_instances(part_name_of_merged_assembly=cube_name)

    def create_mesh_of_part(self, part_name):
        self.model.parts[part_name].seedPart(size=self.params['mesh_size'], minSizeFactor=0.9, deviationFactor=0.1)
        self.model.parts[part_name].generateMesh()

    def merge_mesh_of_instances(self, part_name_of_merged_assembly):
        _instances = self.root_assembly.instances.values()
        self.root_assembly.InstanceFromBooleanMerge(name=part_name_of_merged_assembly, instances=_instances,
                                                    originalInstances=DELETE, mergeNodes=BOUNDARY_ONLY,
                                                    nodeMergingTolerance=1e-06, domain=MESH)

    def create_material(self, material_name, density, engineering_constants):

        self.model.Material(name=material_name)
        self.model.materials[material_name].Density(table=((density,),))
        self.model.materials[material_name].Elastic(type=ENGINEERING_CONSTANTS, table=(engineering_constants,))

    def assign_section_to_elements_of_part_by_bounding_box(self, part_name, material_name, section_name,
                                                           bound_definition):
        bounded_elements = self.model.parts[part_name].elements.getByBoundingBox(**bound_definition)
        bounded_region = regionToolset.Region(elements=bounded_elements)
        self.model.HomogeneousSolidSection(material=material_name, name=section_name, thickness=None)
        self.model.parts[part_name].SectionAssignment(region=bounded_region, sectionName=section_name,
                                                      offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',
                                                      thicknessAssignment=FROM_SECTION)
        self.model.parts[part_name].MaterialOrientation(additionalRotationType=ROTATION_NONE, axis=AXIS_1,
                                                        fieldName='', localCsys=None, orientationType=GLOBAL,
                                                        region=bounded_region, stackDirection=STACK_3)

    def create_set_of_part_by_bounding_box(self, part_name, set_name, bound_definition):
        self.model.parts[part_name].Set(name=set_name,
                                        nodes=self.model.parts[part_name].nodes.getByBoundingBox(**bound_definition))

    def create_set_by_bounding_box(self, instance_name, set_name, bound_definition):
        self.root_assembly.Set(name=set_name, nodes=self.root_assembly.instances[instance_name].nodes.getByBoundingBox(
                                   **bound_definition))

    def set_encastre(self, bc_name, set_name, step_name):
        self.model.EncastreBC(name=bc_name, createStepName=step_name,
                              localCsys=None, region=self.root_assembly.sets[set_name])

    def set_displacement(self, bc_name, set_name, step_name, displacement):
        self.model.DisplacementBC(name=bc_name, createStepName=step_name,
                                  amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None,
                                  region=self.root_assembly.sets[set_name], **displacement)

    def set_boundary_condition(self, symmetry_direction, set_name):
        if symmetry_direction == 'x':
            self.model.XsymmBC(createStepName='Initial', localCsys=None, name='x_sym',
                               region=self.root_assembly.sets[set_name])
        elif symmetry_direction == 'y':
            self.model.YsymmBC(createStepName='Initial', localCsys=None, name='y_sym',
                               region=self.root_assembly.sets[set_name])
        elif symmetry_direction == 'z':
            self.model.ZsymmBC(createStepName='Initial', localCsys=None, name='z_sym',
                               region=self.root_assembly.sets[set_name])
        else:
            raise ValueError

    def create_step(self, step_name, previous_step, step_type):
        if step_type == 'modal':
            self.model.FrequencyStep(name=step_name, previous=previous_step,
                                     limitSavedEigenvectorRegion=None, numEigen=12)
        elif step_type == 'compression':
            self.model.StaticStep(initialInc=0.001, maxInc=0.1, maxNumInc=10000, minInc=1e-12,
                                  name=step_name, previous=previous_step)
        else:
            raise ValueError

    def create_output_requests(self, step_name, history_output_name, set_name,
                               field_outputs, history_outputs):
        self.model.fieldOutputRequests['F-Output-1'].setValues(variables=field_outputs)
        self.model.HistoryOutputRequest(createStepName=step_name, name=history_output_name, rebar=EXCLUDE,
                                        region=self.root_assembly.sets[set_name], sectionPoints=DEFAULT,
                                        variables=history_outputs)

    def create_reference_point_and_set(self, rp_name, rp_coordinate):
        rp_id = self.root_assembly.ReferencePoint(point=rp_coordinate).id
        self.root_assembly.Set(name=rp_name, referencePoints=(self.root_assembly.referencePoints[rp_id],))
        return rp_id

    def create_coupling(self, rp_set_name, surface_set_name, constraint_name):
        self.model.Coupling(alpha=0.0, controlPoint=self.root_assembly.sets[rp_set_name], couplingType=KINEMATIC,
                            influenceRadius=WHOLE_SURFACE, localCsys=None, name=constraint_name,
                            surface=self.root_assembly.sets[surface_set_name],
                            u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    def allow_self_contact(self, instance_name, whole_bound, step_name):
        elements = self.root_assembly.instances[instance_name].elements.getByBoundingBox(**whole_bound)
        surface = self.root_assembly.Surface(name='surface',
                                             face1Elements=elements, face2Elements=elements, face3Elements=elements,
                                             face4Elements=elements, face5Elements=elements, face6Elements=elements)
        self.model.ContactProperty('IntProp-1')
        self.model.interactionProperties['IntProp-1'].NormalBehavior(
            allowSeparation=ON, constraintEnforcementMethod=DEFAULT, pressureOverclosure=HARD)
        self.model.SelfContactStd(contactTracking=ONE_CONFIG,
                                  createStepName=step_name, interactionProperty='IntProp-1', name='Int-1',
                                  surface=surface, thickness=ON)

    def create_job(self, job_name, num_cpus, num_gpus, run):
        mdb.Job(name=job_name, model=self.model.name, description='',
                type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
                scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1,
                multiprocessingMode=DEFAULT, numCpus=num_cpus, numGPUs=num_gpus)
        if run:
            mdb.jobs[job_name].submit(consistencyChecking=OFF)


def random_array(shape, probability):
    from functools import reduce
    return np.random.choice([1, 0], size=reduce(lambda x, y: x * y, shape),
                            p=[probability, 1 - probability]).reshape(shape)


# def make_333(merged_part_name):
#     for ix in range(3):
#         for iy in range(3):
#             for iz in range(3):
#                 instance_name = '{}-{}-{}-{}'.format(CUBE_NAME, ix, iy, iz)
#                 ins = root_assembly.Instance(name=instance_name, part=root_model.parts[CUBE_NAME], dependent=ON)
#                 ins.translate(vector=(ix * CUBE_X_SIZE, iy * CUBE_Y_SIZE, iz * CUBE_Z_SIZE))
#     root_assembly.InstanceFromBooleanMerge(name=merged_part_name, instances=root_assembly.instances.values(),
#                                            originalInstances=DELETE, mergeNodes=BOUNDARY_ONLY,
#                                            nodeMergingTolerance=1e-06, domain=MESH)


def quaver_to_full(quaver):
    quarter = np.concatenate((np.flip(quaver, axis=0), quaver), axis=0)
    half = np.concatenate((np.flip(quarter, axis=1), quarter), axis=1)
    full = np.concatenate((np.flip(half, axis=2), half), axis=2)
    return np.swapaxes(full, axis1=0, axis2=2)


def bound_setter(whole_bound, option):
    """
    Create dictionary of bound limits used for bounding box
    :param whole_bound: whole bound dictionary for bounding box, six keys in dictionary
    :param option: This can be either 'xMin', 'yMin', 'zMin', 'xMax', 'yMax', 'zMax'. If 'xMin',
     bound for x=0 will be returned.
    :return: a dictionary for bounding box
    """
    bound = whole_bound.copy()
    if 'Min' in option:
        bound[option[0] + 'Max'] = bound[option[0] + 'Min']
    elif 'Max' in option:
        bound[option[0] + 'Min'] = bound[option[0] + 'Max']
    else:
        pass
    return bound


def run_analysis(model_name, topo_arr, voxel_name, voxel_unit_length, cube_name,
                 analysis_mode, material_properties, full):
    topo_arr = quaver_to_full(topo_arr) if full else topo_arr.copy()
    cube_x_voxels, cube_y_voxels, cube_z_voxels = topo_arr.shape
    cube_x_size = voxel_unit_length * cube_x_voxels
    cube_y_size = voxel_unit_length * cube_y_voxels
    cube_z_size = voxel_unit_length * cube_z_voxels
    whole_bound = {'xMin': 0., 'yMin': 0., 'zMin': 0., 'xMax': cube_x_size, 'yMax': cube_y_size, 'zMax': cube_z_size}
    bounds = {option: bound_setter(whole_bound=whole_bound, option=option) for option in whole_bound.keys()}
    rp_coordinates = {'RP-x': (1.05 * cube_x_size, cube_y_size / 2, cube_z_size / 2),
                      'RP-y': (cube_x_size / 2, 1.05 * cube_y_size, cube_z_size / 2),
                      'RP-z': (cube_x_size / 2, cube_y_size / 2, 1.05 * cube_z_size)}

    with MyModel(model_name='Model-{}'.format(model_name), params=parameters) as mm:
        material_name = material_properties['material_name']
        mm.create_voxel_part(voxel_name=voxel_name)
        mm.create_mesh_of_part(part_name=voxel_name)
        mm.create_cube_part(voxel_name=voxel_name, cube_name=cube_name, topo_arr=topo_arr)
        mm.create_material(**material_properties)
        for option, bound in bounds.items():
            mm.create_set_of_part_by_bounding_box(part_name=cube_name, set_name=option, bound_definition=bound)
        rp_id_dict = {rpn: mm.create_reference_point_and_set(rp_name=rpn, rp_coordinate=rpc)
                      for rpn, rpc in rp_coordinates.items()}
        mm.create_coupling(rp_set_name='RP-y', surface_set_name='{}-1.{}'.format(cube_name, 'yMax'),
                           constraint_name='coupling')

        for symmetry_diction, mirror_set_name in (('x', 'xMin'), ('y', 'yMin'), ('z', 'zMin')):
            mm.set_boundary_condition(symmetry_direction=symmetry_diction,
                                      set_name='{}-1.{}'.format(cube_name, mirror_set_name))
        mm.assign_section_to_elements_of_part_by_bounding_box(part_name=cube_name, material_name=material_name,
                                                              section_name=material_name + '-section',
                                                              bound_definition=whole_bound)
        if analysis_mode == 'modal':
            mm.create_step(step_name=analysis_mode + '-step', previous_step='Initial', step_type=analysis_mode)
            mm.set_encastre(bc_name='encastre_bottom', set_name='{}-1.{}'.format(cube_name, 'yMin'),
                            step_name='Initial')
        elif analysis_mode == 'compression':
            analysis_step_name = analysis_mode + '-step'
            mm.create_step(step_name=analysis_step_name, previous_step='Initial', step_type=analysis_mode)
            mm.set_displacement(bc_name='displacement',
                                set_name='{}-1.{}'.format(cube_name, 'yMax'),
                                step_name=analysis_step_name, displacement={'u1': 0, 'u2': -0.5, 'u3': 0,
                                                                            'ur1': 0, 'ur2': 0, 'ur3': 0})
            mm.set_displacement(bc_name='fix_x',
                                set_name='{}-1.{}'.format(cube_name, 'xMin'),
                                step_name=analysis_step_name, displacement={'u1': 0})
            mm.set_displacement(bc_name='fix_y',
                                set_name='{}-1.{}'.format(cube_name, 'yMin'),
                                step_name=analysis_step_name, displacement={'u2': 0})
            mm.set_displacement(bc_name='fix_z',
                                set_name='{}-1.{}'.format(cube_name, 'zMin'),
                                step_name=analysis_step_name, displacement={'u3': 0})
            # mm.allow_self_contact(instance_name=cube_name + '-1', whole_bound=whole_bound, step_name=analysis_step_name)
            mm.create_output_requests(step_name=analysis_step_name, history_output_name='H-Output',
                                      set_name='RP-y',
                                      field_outputs=('S', 'U', 'RF', 'IVOL', 'MISESMAX'),
                                      history_outputs=('U1', 'U2', 'U3', 'RF1', 'RF2', 'RF3', 'ALLIE'))
        else:
            raise ValueError

        # mm.create_job(job_name='Job-{}'.format(model_name), num_cpus=1, num_gpus=0, run=True)


if __name__ == '__main__':
    path = r'C:\pythoncode\AuxeticMOP\abaqus data\topo_parent_1.csv'
    material = {
        'material_name': 'resin',
        'density': 1.2e-09,
        'engineering_constants': (1500, 1200, 1500, 0.35, 0.35, 0.35, 450, 550, 450)
    }
    parameters = {
        'end_pop': 100,
        'lx': 5,
        'ly': 5,
        'lz': 5,
        'unit_l': 3.0,
        'mesh_size': 1.5
    }
    topos_from_csv = np.swapaxes(np.genfromtxt(path, delimiter=',', dtype=int).reshape((
        parameters['end_pop'], parameters['lx'], parameters['ly'], parameters['lz'])), axis1=1, axis2=3)
    for model_num, topo in enumerate(topos_from_csv, start=1):
        run_analysis(model_name='1-{}'.format(model_num), analysis_mode='compression',
                     topo_arr=topo, voxel_unit_length=parameters['unit_l'], full=False,
                     material_properties=material, voxel_name='voxel', cube_name='cube')
        break
