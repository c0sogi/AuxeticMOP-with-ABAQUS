import regionToolset
from abaqus import *
from abaqusConstants import *
from driverUtils import executeOnCaeStartup
from odbAccess import openOdb
import numpy as np
import pickle
import os
from datetime import datetime
import threading
import socket
import struct
import json
from sys import version_info
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
try:
    from Queue import Queue
except ImportError:
    from queue import Queue

executeOnCaeStartup()
HOST = 'localhost'
PORT = 12345


class Client:
    def __init__(self, host, port, option, connect):
        self.host = host
        self.port = port
        self.option = option
        self.q = Queue()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.is_alive = True
        self._default_packet_size = 1024
        self._header_format = '>I'
        self._header_bytes = 4
        if connect:
            self.connect()

    def connect(self):
        if version_info.major >= 3:
            new_th = threading.Thread(target=self._thread_recv, args=(self.client_socket, self.option), daemon=True)
        else:
            new_th = threading.Thread(target=self._thread_recv, args=(self.client_socket, self.option))
            new_th.setDaemon(True)
        self.client_socket.connect((self.host, self.port))
        print('[{}] Connected to {}:{}'.format(datetime.now(), self.host, self.port))
        new_th.start()

    def send(self, data):
        if self.option == 'pickle':
            serialized_data = pickle.dumps(data, protocol=2)
        else:
            serialized_data = json.dumps(data).encode()
        while True:
            try:
                self.client_socket.sendall(struct.pack(self._header_format, len(serialized_data)))
                self.client_socket.sendall(serialized_data)
                print('[{}] A data sent'.format(datetime.now()))
                break
            except Exception as send_error:
                print('Sending data failed, trying to reconnect to server: ', send_error)
                self.connect()

    def recv(self):
        return self.q.get()

    def close(self):
        self.client_socket.close()

    def _thread_recv(self, client_socket, option):
        while True:
            try:  # Trying to receive a data and decode it
                data_size = struct.unpack(self._header_format, client_socket.recv(self._header_bytes))[0]
                remaining_payload_size = data_size
                packets = b''
                while remaining_payload_size != 0:
                    packets += client_socket.recv(remaining_payload_size)
                    remaining_payload_size = data_size - len(packets)
                try:  # Trying to decode received data
                    if option == 'json':
                        received_data = json.loads(packets.decode())
                    else:
                        if version_info.major >= 3:
                            received_data = pickle.loads(packets, encoding='bytes')
                        else:
                            received_data = pickle.loads(packets)
                    print('[{}] Received data: {}'.format(datetime.now(), received_data))
                    self.q.put(received_data)
                except Exception as e2:  # Decoding is failed
                    print('[{}] Loading received data failure: {}'.format(datetime.now(), e2))
                    continue
            except Exception as e1:  # Connection is lost
                print('[{}] Error: {}'.format(datetime.now(), e1))
                break
        self.is_alive = False
        print('<!> Connection dead')


class JobLogFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.text = tk.Text(self, height=50, width=100)
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)


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
        del mdb.models[self.model.name]

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

    def allow_self_contact(self, instance_name, step_name):
        elements = self.root_assembly.instances[instance_name].elements.getExteriorFaces()
        _surface = self.root_assembly.Surface(name='ExteriorSurface', side1Elements=elements)
        _interaction_property_name = 'SelfContactProp'
        _interaction_name = 'SelfContact'
        self.model.ContactProperty(_interaction_property_name)
        self.model.interactionProperties[_interaction_property_name].NormalBehavior(
            allowSeparation=ON, constraintEnforcementMethod=DEFAULT, pressureOverclosure=HARD)
        self.model.SelfContactStd(name=_interaction_name, contactTracking=ONE_CONFIG, createStepName=step_name,
                                  interactionProperty=_interaction_property_name, surface=_surface, thickness=ON)

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
            mdb.jobs[job_name].waitForCompletion()


# Function for Python 2
def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
    return dict(map(ascii_encode, pair) for pair in data.items())


def open_job_log():
    _root = tk.Tk()
    _root.title('Abaqus control log')
    _frame = JobLogFrame(_root)
    _frame.pack(fill="both", expand=True)
    _new_thread = threading.Thread(target=_root.mainloop)
    _new_thread.setDaemon(True)
    _new_thread.start()
    return _frame


def save_log(message, job_log_frame):
    _now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    _message = '[{}]{}\n'.format(_now, message)
    print(_message)
    with open('log.txt', mode='a') as f_log:
        f_log.write(_message)
    job_log_frame.text.insert('end', _message)
    job_log_frame.text.see('end')


def dump_pickled_dict_data(file_name, key, to_dump, mode):
    if mode == 'a' and os.path.isfile(file_name):
        with open(file_name, mode='rb') as f:
            dict_data = pickle.load(f)
        dict_data.update({key: to_dump})
    else:
        dict_data = {key: to_dump}
    with open(file_name, mode='wb') as f:
        pickle.dump(dict_data, f)


def load_pickled_dict_data(file_name):
    with open(file_name, mode='rb') as f:
        dict_data = pickle.load(f)
    return dict_data


def random_array(shape, probability):
    from functools import reduce
    return np.random.choice([1, 0], size=reduce(lambda x, y: x * y, shape),
                            p=[probability, 1 - probability]).reshape(shape)


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


def export_outputs(model_name, step_name, rp_name):
    gen, entity = map(int, model_name.split('-'))
    history_output_file_header = 'HistoryOutput_offspring'
    field_output_file_header = 'FieldOutput_offspring'
    exported_field_outputs = {
        'displacement': {'xMax': np.ndarray, 'yMax': np.ndarray, 'zMax': np.ndarray},
        'rotation': np.ndarray,
        'reaction_force': np.ndarray,
        'mises_stress': {'max': float, 'min': float, 'average': float}
    }
    exported_history_output_properties = ('U1', 'U2', 'U3', 'RF1', 'RF2', 'RF3')
    try:
        odb = openOdb('Job-{}.odb'.format(model_name))
        try:
            # Field outputs
            field_outputs = {
                'displacement': odb.steps[step_name].frames[-1].fieldOutputs['U'],
                'rotation': odb.steps[step_name].frames[-1].fieldOutputs['UR'],
                'reaction_force': odb.steps[step_name].frames[-1].fieldOutputs['RF'],
                'stress': odb.steps[step_name].frames[-1].fieldOutputs['S']
            }
            displacement_of_node_sets = {
                face: field_outputs['displacement'].getSubset(region=odb.rootAssembly.nodeSets[face.upper()]).values
                for face in ('xMax', 'yMax', 'zMax')
            }
            for face, displacement_of_node_set in displacement_of_node_sets.items():
                exported_field_outputs['displacement'][face] = np.average(
                    np.array([data.data for data in displacement_of_node_set]), axis=0)
            exported_field_outputs['reaction_force'] = field_outputs['reaction_force'].getSubset(
                region=odb.rootAssembly.nodeSets[rp_name.upper()]).values[0].data
            exported_field_outputs['rotation'] = field_outputs['rotation'].getSubset(
                region=odb.rootAssembly.nodeSets[rp_name.upper()]).values[0].data
            _values = [stress.mises for stress in field_outputs['stress'].values]
            exported_field_outputs['mises_stress']['max'] = max(_values)
            exported_field_outputs['mises_stress']['min'] = min(_values)
            exported_field_outputs['mises_stress']['average'] = np.average(_values)
            dump_pickled_dict_data(file_name='{}_{}'.format(field_output_file_header, str(gen)),
                                   key=entity, to_dump=exported_field_outputs, mode='a')

            # History outputs
            history_outputs = odb.steps[step_name].historyRegions['Node ASSEMBLY.1'].historyOutputs
            exported_history_outputs = {prop: np.array(history_outputs[prop].data)
                                        for prop in exported_history_output_properties}
            dump_pickled_dict_data(file_name='{}_{}'.format(history_output_file_header, str(gen)),
                                   key=entity, to_dump=exported_history_outputs, mode='a')

        except Exception as e2:
            print('ODB output reading failed: ', e2)
        finally:
            odb.close()
    except Exception as e1:
        print('There is no ODB file: ', e1)


def run_analysis(params, model_name, topo_arr, voxel_name, voxel_unit_length, cube_name,
                 analysis_mode, material_properties, full, displacement=None):
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

    with MyModel(model_name='Model-{}'.format(model_name), params=params) as mm:
        material_name = material_properties['material_name']
        mm.create_voxel_part(voxel_name=voxel_name)
        mm.create_mesh_of_part(part_name=voxel_name)
        mm.create_cube_part(voxel_name=voxel_name, cube_name=cube_name, topo_arr=topo_arr)
        mm.create_material(**material_properties)
        mm.assign_section_to_elements_of_part_by_bounding_box(part_name=cube_name, material_name=material_name,
                                                              section_name=material_name + '-section',
                                                              bound_definition=whole_bound)
        for option, bound in bounds.items():
            mm.create_set_by_bounding_box(instance_name=cube_name + '-1', set_name=option, bound_definition=bound)
        for rp_name, rp_coordinate in rp_coordinates.items():
            mm.create_reference_point_and_set(rp_name=rp_name, rp_coordinate=rp_coordinate)
        for symmetry_diction, boundary_set_name in (('x', 'xMin'), ('y', 'yMin'), ('z', 'zMin')):
            mm.set_boundary_condition(symmetry_direction=symmetry_diction, set_name=boundary_set_name)
        if analysis_mode == 'modal':
            mm.create_step(step_name=analysis_mode + '-step', previous_step='Initial', step_type=analysis_mode)
            mm.set_encastre(bc_name='encastre_bottom', set_name='yMin', step_name='Initial')
        elif analysis_mode == 'compression':
            analysis_step_name = analysis_mode + '-step'
            mm.create_step(step_name=analysis_step_name, previous_step='Initial', step_type=analysis_mode)
            mm.allow_self_contact(instance_name=cube_name + '-1', step_name=analysis_step_name)
            mm.create_coupling(rp_set_name='RP-y', surface_set_name='yMax', constraint_name='coupling')
            mm.set_displacement(bc_name='displacement',
                                set_name='RP-y',
                                step_name=analysis_step_name, displacement=displacement)
            mm.create_output_requests(step_name=analysis_step_name, history_output_name='H-Output', set_name='RP-y',
                                      field_outputs=('S', 'U', 'RF', 'IVOL', 'MISESMAX'),
                                      history_outputs=('U1', 'U2', 'U3', 'RF1', 'RF2', 'RF3', 'ALLIE'))
        else:
            raise ValueError
        mm.root_assembly.regenerate()
        mm.create_job(job_name='Job-{}'.format(model_name),
                      num_cpus=params['n_cpus'], num_gpus=params['n_gpus'], run=True)
        export_outputs(model_name=model_name, step_name=analysis_step_name, rp_name='RP-y')


if __name__ == '__main__':
    client = Client(host=HOST, port=PORT, option='json', connect=True)
    frame = open_job_log()
    save_log('connected to {}:{}'.format(PORT, HOST), job_log_frame=frame)
    while True:
        parameters = client.recv()
        parameters = ascii_encode_dict(parameters)
        if parameters['exit_abaqus']:
            break
        material_property_definitions = {
            'material_name': parameters['material_name'],
            'density': parameters['density'],
            'engineering_constants': parameters['engineering_constants']
        }
        start_topology_from = parameters['start_topology_from']
        topologies_file_name = parameters['topologies_file_name']
        topologies_key = parameters['topologies_key']
        topologies = load_pickled_dict_data(topologies_file_name)[topologies_key]
        gen_num = topologies_file_name.split('_')[-1]
        for entity_num, topology in enumerate(topologies, start=1):
            if entity_num < start_topology_from:
                continue
            run_analysis(model_name='{}-{}'.format(gen_num, entity_num), analysis_mode='compression',
                         topo_arr=topology, voxel_unit_length=parameters['unit_l'], full=False, params=parameters,
                         material_properties=material_property_definitions, voxel_name='voxel', cube_name='cube',
                         displacement={'u1': 0, 'u2': parameters['dis_y'], 'u3': 0, 'ur1': 0, 'ur2': 0, 'ur3': 0})
            save_log('Created Job{}-{}.odb'.format(gen_num, entity_num), job_log_frame=frame)
        client.send('[{}] Generation {} finished!'.format(datetime.now(), gen_num))
