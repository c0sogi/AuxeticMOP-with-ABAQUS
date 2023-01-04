from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import *
executeOnCaeStartup()

# execfile('abaqus_scripts.py', __main__.__dict__)
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from odbAccess import *

import os
import numpy as np
import pickle
from datetime import datetime
from time import sleep
import threading
try:
    import Tkinter as tk
except:
    import tkinter as tk

# Load parameter dictionary from PARAMS
with open('./PARAMS', mode='rb') as f_params:
    PARAMS = pickle.load(f_params)
os.remove('./PARAMS')

# Unused variables
model = 'original'

# Load variables from PARAMS
setPath = PARAMS['setPath']
os.chdir(setPath)
mode = PARAMS['mode']
restart_pop = PARAMS['restart_pop']
ini_pop = PARAMS['ini_pop']
end_pop = PARAMS['end_pop']
divide_number = PARAMS['divide_number']
unit_l = PARAMS['unit_l']
lx = PARAMS['lx']
ly = PARAMS['ly']
lz = PARAMS['lz']
mesh_size = PARAMS['mesh_size']
dis_y = PARAMS['dis_y']
density = PARAMS['density']
material_modulus = PARAMS['material_modulus']
poissons_ratio = PARAMS['poissons_ratio']
MAXRF22 = PARAMS['MAXRF22']
penalty_coefficient = PARAMS['penalty_coefficient']
n_cpus = PARAMS['n_cpus']
n_gpus = PARAMS['n_gpus']

# Define another parameters from PARAMS
unit_l_half = unit_l * 0.5
unit_lx_total = lx * unit_l
unit_ly_total = ly * unit_l
unit_lz_total = lz * unit_l


class LogFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.text = tk.Text(self, height=50, width=100)
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)


def open_log_window():
    root =tk.Tk()
    root.title('Abaqus control log')
    fr = LogFrame(root)
    fr.pack(fill="both", expand=True)
    th_log = threading.Thread(target=root.mainloop)
    th_log.start()
    sleep(1)
    return fr


def now_s():
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S')


def array_to_csv(path, arr, dtype, mode, save_as_int=False):
    if mode == 'a' and os.path.isfile(path):
        previous_arr = csv_to_array(path, dtype=dtype)
        print('[array_to_csv] append shape: ', previous_arr.shape, arr.shape)
        arr = np.vstack((previous_arr, arr))
    fmt = '%i' if save_as_int else '%.18e'
    np.savetxt(path, arr, delimiter=',', fmt=fmt)


def csv_to_array(path, dtype):
    return np.genfromtxt(path, delimiter=',', dtype=dtype)


def save_log(log_message, frame):
    print(log_message)
    with open('./log.txt', mode='a') as f_log:
        f_log.write(log_message + '\n')
    frame.text.insert('end', log_message + '\n')
    frame.text.see('end')


def abaqus_cad(offspring, m, rt, q):  # offspring == 4dimensional numpy array
    m.ConstrainedSketch(name='__profile__', sheetSize=200.0)
    m.sketches['__profile__'].rectangle(point1=(0.0, 0.0),
                                        point2=(unit_l, unit_l))

    m.Part(dimensionality=THREE_D, name='Part-1', type=DEFORMABLE_BODY)
    m.parts['Part-1'].BaseSolidExtrude(depth=unit_l, sketch=m.sketches['__profile__'])
    del m.sketches['__profile__']

    voxelnum = np.sum(offspring[q - 1])
    voxelnum = int(voxelnum)
    rt.DatumCsysByDefault(CARTESIAN)
    rt.Instance(dependent=ON, name='Part-1', part=m.parts['Part-1'])
    rt.LinearInstancePattern(direction1=(1.0, 0.0, 0.0), direction2=(0.0, 1.0, 0.0),
                             instanceList=('Part-1',),
                             number1=int(voxelnum),
                             number2=1, spacing1=0.0, spacing2=0.0)

    n = 0
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                if offspring[q - 1][k][j][i] == 1:
                    if n == 0:
                        rt.translate(instanceList=('Part-1',), vector=(unit_l * i, unit_l * j, unit_l * k))
                    else:
                        rt.translate(instanceList=('Part-1-lin-%d-1' % (n + 1),),
                                     vector=(i * unit_l, j * unit_l, k * unit_l))
                    n = n + 1


def abaqus_cad_ones(topologys, m, rt):  # topologys == 3dimensional numpy array

    a = unit_l / divide_number

    m.ConstrainedSketch(name='__profile__', sheetSize=200.0)

    m.sketches['__profile__'].rectangle(point1=(0.0, 0.0),

                                        point2=(a, a))

    m.Part(dimensionality=THREE_D, name='Part-1', type=DEFORMABLE_BODY)

    m.parts['Part-1'].BaseSolidExtrude(depth=a, sketch=m.sketches['__profile__'])

    del m.sketches['__profile__']

    voxelnum = np.sum(topologys)

    voxelnum = int(voxelnum)

    rt.DatumCsysByDefault(CARTESIAN)

    rt.Instance(dependent=ON, name='Part-1', part=m.parts['Part-1'])

    rt.LinearInstancePattern(direction1=(1.0, 0.0, 0.0), direction2=(0.0, 1.0, 0.0),

                             instanceList=('Part-1',),

                             number1=int(voxelnum),

                             number2=1, spacing1=0.0, spacing2=0.0)

    n = 0

    for i in range(lx):

        for j in range(ly):

            for k in range(lz):

                if topologys[k][j][i] == 1:

                    if n == 0:

                        rt.translate(instanceList=('Part-1',), vector=(unit_l * i, unit_l * j, unit_l * k))

                    else:

                        rt.translate(instanceList=('Part-1-lin-%d-1' % (n + 1),),
                                     vector=(i * unit_l, j * unit_l, k * unit_l))

                    n = n + 1


def abaqus_merge3(m, rt):
    SingleInstances_List = rt.instances.keys()

    c = [0 for i in range(len(SingleInstances_List))]

    c[0] = rt.instances['Part-1']

    for i in range(1, len(SingleInstances_List)):
        c[i] = rt.instances['Part-1-lin-%d-1' % (i + 1)]

    rt.InstanceFromBooleanMerge(name='Merge-1', instances=c, keepIntersections=ON,

                                domain=GEOMETRY, originalInstances=SUPPRESS)

    del m.parts['Part-1']


def abaqus_merge4(split, m, rt):
    SingleInstances_List = rt.instances.keys()
    SingleInstances_Num = len(SingleInstances_List)
    one_ten = int(SingleInstances_Num // split)

    for j in range(split):

        if j == 0:
            c = [0 for i in range(one_ten)]
            c[-1] = rt.instances['Part-1']
            for i in range(one_ten - 1):
                c[i] = rt.instances['Part-1-lin-%d-1' % (i + 2)]
            rt.InstanceFromBooleanMerge(name='Merge-%d' % (j + 2), instances=c, keepIntersections=ON, domain=GEOMETRY,
                                        originalInstances=SUPPRESS)

        elif 0 < j < split - 1:
            c = [0 for i in range(one_ten + 1)]
            c[-1] = rt.instances['Merge-%d-1' % (j + 1)]
            for i in range(one_ten):
                c[i] = rt.instances['Part-1-lin-%d-1' % ((j * one_ten) + i + 1)]
            rt.InstanceFromBooleanMerge(name='Merge-%d' % (j + 2), instances=c, keepIntersections=ON, domain=GEOMETRY,
                                        originalInstances=SUPPRESS)
            del m.parts['Merge-%d' % (j + 1)]

        else:

            if SingleInstances_Num % split == 0:

                c = [0 for i in range(one_ten + 1)]

                c[-1] = rt.instances['Merge-%d-1' % (j + 1)]

                for i in range(one_ten):
                    c[i] = rt.instances['Part-1-lin-%d-1' % ((j * one_ten) + i + 1)]

                rt.InstanceFromBooleanMerge(name='Merge-1', instances=c, keepIntersections=ON,

                                            domain=GEOMETRY, originalInstances=SUPPRESS)

                del m.parts['Merge-%d' % (j + 1)]

            else:

                c = [0 for i in range(one_ten + 1 + (SingleInstances_Num % split))]

                c[-1] = rt.instances['Merge-%d-1' % (j + 1)]

                for i in range(one_ten + (SingleInstances_Num % split)):
                    c[i] = rt.instances['Part-1-lin-%d-1' % ((j * one_ten) + i + 1)]

                rt.InstanceFromBooleanMerge(name='Merge-1', instances=c, keepIntersections=ON,

                                            domain=GEOMETRY, originalInstances=SUPPRESS)

                del m.parts['Merge-%d' % (j + 1)]

            # def abaqus_merge():


def abaqus_mesh(m):
    m.parts['Merge-1'].seedPart(deviationFactor=0.1, minSizeFactor=0.90, size=mesh_size)
    m.parts['Merge-1'].generateMesh()


def abaqus_step(m, rt):
    ## step generation

    rt.regenerate()

    m.StaticStep(initialInc=0.001, maxInc=0.1, maxNumInc=10000, minInc=1e-12, name='Step-1', previous='Initial')

    ## Define Reference point

    r = rt.referencePoints

    RP2id = rt.ReferencePoint(point=(0.5 * unit_lx_total, 1.1 * unit_ly_total, 0.5 * unit_lz_total)).id

    RP2 = (r[RP2id],)

    rt.Set(name='RP2', referencePoints=RP2)

    RP1id = rt.ReferencePoint(point=(1.1 * unit_lx_total, 0.5 * unit_ly_total, 0.5 * unit_lz_total)).id

    RP1 = (r[RP1id],)

    rt.Set(name='RP1', referencePoints=RP1)

    RP3id = rt.ReferencePoint(point=(0.5 * unit_lx_total, 0.5 * unit_ly_total, 1.1 * unit_lz_total)).id

    RP3 = (r[RP3id],)

    rt.Set(name='RP3', referencePoints=RP3)

    ## Output request

    m.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'RF', 'IVOL', 'MISESMAX'))

    m.HistoryOutputRequest(createStepName='Step-1', name='RP2_H-Output',

                           rebar=EXCLUDE, region=rt.sets['RP2'],

                           sectionPoints=DEFAULT, variables=('U1', 'U2', 'U3', 'RF1', 'RF2', 'RF3', 'ALLIE'))

    ## Step and job setting

    m.steps['Step-1'].setValues(initialInc=0.0001, timePeriod=1.0)


def abaqus_jobsetting(w, q):
    if mode == 'GA' or mode == 'Random' or mode == 'None':
        mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,

                explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,

                memory=90, memoryUnits=PERCENTAGE, model='Model-%d' % (q), modelPrint=OFF,

                multiprocessingMode=DEFAULT, name='Job-%d-%d' % (w, q), nodalOutputPrecision=SINGLE,

                numCpus=n_cpus, numDomains=n_cpus, numGPUs=n_gpus, queue=None, resultsFormat=ODB,

                scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)

    if mode == 'Gaussian':
        if model == 'original':
            mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,

                    explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,

                    memory=90, memoryUnits=PERCENTAGE, model='Model-%d-%d_original' % (w, q), modelPrint=OFF,

                    multiprocessingMode=DEFAULT, name='Job-%d-%d_original' % (w, q), nodalOutputPrecision=SINGLE,

                    numCpus=n_cpus, numDomains=n_cpus, numGPUs=n_gpus, queue=None, resultsFormat=ODB,

                    scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)

        if model == 'gaussian':
            mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,

                    explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,

                    memory=90, memoryUnits=PERCENTAGE, model='Model-%d-%d_gaussian' % (w, q), modelPrint=OFF,

                    multiprocessingMode=DEFAULT, name='Job-%d-%d_gaussian' % (w, q), nodalOutputPrecision=SINGLE,

                    numCpus=n_cpus, numDomains=n_cpus, numGPUs=n_gpus, queue=None, resultsFormat=ODB,

                    scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)


def abaqus_material_elastic(modulus, poisson, m, q):
    m.Material(name='Material-1')

    m.materials['Material-1'].Elastic(table=((modulus, poisson),))

    m.HomogeneousSolidSection(material='Material-1', name=

    'Section-1', thickness=None)

    ## Section assignment

    for i in range(lx):

        for j in range(ly):

            for k in range(lz):

                if offspring[q - 1][k][j][i] == 1:
                    m.parts['Merge-1'].SectionAssignment(offset=0.0,

                                                         offsetField='', offsetType=MIDDLE_SURFACE,

                                                         region=Region(cells=m.parts[

                                                             'Merge-1'].cells.findAt(((unit_l * i + unit_l_half,
                                                                                       unit_l * j + unit_l_half,
                                                                                       unit_l * k + unit_l_half),), ))

                                                         , sectionName='Section-1', thicknessAssignment=FROM_SECTION)


def abaqus_material_elastic2(modulus, poisson, m):
    m.Material(name='Material-1')

    m.materials['Material-1'].Elastic(table=((modulus, poisson),))

    m.HomogeneousSolidSection(material='Material-1', name=

    'Section-1', thickness=None)

    ## Section assignment

    m.parts['Merge-1'].SectionAssignment(offset=0.0,

                                         offsetField='', offsetType=MIDDLE_SURFACE,

                                         region=Region(cells=m.parts[

                                             'Merge-1'].cells.getByBoundingBox(-0.5, -0.5, -0.5, unit_lx_total + 0.5,
                                                                               unit_ly_total + 0.5,
                                                                               unit_lz_total + 0.5))

                                         , sectionName='Section-1', thicknessAssignment=FROM_SECTION)


def abaqus_material_anisotropic(offspring, E11, E22, E33, v12, v13, v23, G12, G13, G23, m, q):
    m.Material(name='Material-1')

    m.materials['Material-1'].Elastic(table=((E11, E22, E33, v12, v13, v23, G12, G13, G23),),
                                      type=ENGINEERING_CONSTANTS)

    m.HomogeneousSolidSection(material='Material-1', name=

    'Section-1', thickness=None)

    ## Section assignment

    for i in range(lx):

        for j in range(ly):

            for k in range(lz):

                if offspring[q - 1][k][j][i] == 1:
                    m.parts['Merge-1'].SectionAssignment(offset=0.0,

                                                         offsetField='', offsetType=MIDDLE_SURFACE,

                                                         region=Region(cells=m.parts[

                                                             'Merge-1'].cells.findAt(((unit_l * i + unit_l_half,
                                                                                       unit_l * j + unit_l_half,
                                                                                       unit_l * k + unit_l_half),), ))

                                                         , sectionName='Section-1', thicknessAssignment=FROM_SECTION)

                    m.parts['Merge-1'].MaterialOrientation(additionalRotationType=ROTATION_NONE, axis=AXIS_1,
                                                           fieldName='', localCsys=None, orientationType=GLOBAL,
                                                           region=Region(

                                                               cells=m.parts['Merge-1'].cells.findAt(((
                                                                                                          unit_l * i + unit_l_half,
                                                                                                          unit_l * j + unit_l_half,
                                                                                                          unit_l * k + unit_l_half),), ))

                                                           , stackDirection=STACK_3)


def abaqus_material_elastic_ones(modulus, poisson, m, topo):
    m.Material(name='Material-1')

    m.materials['Material-1'].Elastic(table=((modulus, poisson),))

    m.HomogeneousSolidSection(material='Material-1', name='Section-1', thickness=None)

    ## Section assignment

    for i in range(lx):

        for j in range(ly):

            for k in range(lz):

                if topo[k][j][i] == 1:
                    m.parts['Merge-1'].SectionAssignment(offset=0.0,

                                                         offsetField='', offsetType=MIDDLE_SURFACE,

                                                         region=Region(cells=m.parts[

                                                             'Merge-1'].cells.findAt(((unit_l * i + unit_l_half,
                                                                                       unit_l * j + unit_l_half,
                                                                                       unit_l * k + unit_l_half),), ))

                                                         , sectionName='Section-1', thicknessAssignment=FROM_SECTION)


def abaqus_material_hyper(m, q):
    m.Material(name='Material-1')

    mdb.models['Model-1'].materials['Material-1'].Density(table=((density,),))

    mdb.models['Model-1'].materials['Material-1'].Hyperelastic(materialType=ISOTROPIC, n=3,

                                                               table=((-2562157.38, 2.69096262, 935416.629, 4.73668103,

                                                                       3581854.08, -1.05562694, 0.0, 0.0, 0.0),),

                                                               testData=OFF, type=OGDEN,
                                                               volumetricResponse=VOLUMETRIC_DATA)

    m.HomogeneousSolidSection(material='Material-1', name='Section-1', thickness=None)

    ## Section assignment

    for i in range(lx):

        for j in range(ly):

            for k in range(lz):

                if offspring[q - 1][k][j][i] == 1:
                    m.parts['Merge-1'].SectionAssignment(offset=0.0,

                                                         offsetField='', offsetType=MIDDLE_SURFACE,

                                                         region=Region(cells=m.parts[

                                                             'Merge-1'].cells.findAt(((unit_l * i + unit_l_half,
                                                                                       unit_l * j + unit_l_half,
                                                                                       unit_l * k + unit_l_half),), ))

                                                         , sectionName='Section-1', thicknessAssignment=FROM_SECTION)


def abaqus_BC_ones(m, rt, topo):
    ##self-contact behavior

    m.ContactProperty('IntProp-1')

    m.interactionProperties['IntProp-1'].NormalBehavior(

        allowSeparation=ON, constraintEnforcementMethod=DEFAULT, pressureOverclosure=HARD)

    rt.Surface(name='Surf-1', side1Faces=

    rt.instances['Merge-1-1'].faces.getByBoundingBox(-0.5, -0.5, -0.5, unit_lx_total + 0.5, unit_ly_total + 0.5,
                                                     unit_lz_total + 0.5))

    m.SelfContactStd(contactTracking=ONE_CONFIG,

                     createStepName='Step-1', interactionProperty='IntProp-1', name='Int-1',

                     surface=rt.surfaces['Surf-1'], thickness=ON)

    ##Boundary conditions

    it = rt.instances['Merge-1-1']

    FaceX0 = []

    FaceX = []

    FaceY0 = []

    FaceY = []

    FaceZ0 = []

    FaceZ = []

    ## select all Face

    for i in range(lx):

        for j in range(ly):

            for k in range(lz):

                if i == 0 and topo[k][j][i] == 1:
                    FaceX0 = FaceX0 + [it.faces.findAt((0, unit_l * j + unit_l_half, unit_l * k + unit_l_half), )]

                if i == lx - 1 and topo[k][j][i] == 1:  # correction: a,b,c >> lx,ly,lz

                    FaceX = FaceX + [
                        it.faces.findAt((unit_lx_total, unit_l * j + unit_l_half, unit_l * k + unit_l_half), )]

                if j == 0 and topo[k][j][i] == 1:
                    FaceY0 = FaceY0 + [it.faces.findAt((unit_l * i + unit_l_half, 0, unit_l * k + unit_l_half), )]

                if j == ly - 1 and topo[k][j][i] == 1:
                    FaceY = FaceY + [
                        it.faces.findAt((unit_l * i + unit_l_half, unit_ly_total, unit_l * k + unit_l_half), )]

                if k == 0 and topo[k][j][i] == 1:
                    FaceZ0 = FaceZ0 + [it.faces.findAt((unit_l * i + unit_l_half, unit_l * j + unit_l_half, 0), )]

                if k == lz - 1 and topo[k][j][i] == 1:
                    FaceZ = FaceZ + [
                        it.faces.findAt((unit_l * i + unit_l_half, unit_l * j + unit_l_half, unit_lz_total), )]

    FX = []

    FX0 = []

    FY = []

    FY0 = []

    FZ = []

    FZ0 = []

    for i in range(0, len(FaceY)):  # AttributeError: 'NoneType' object has no attribute 'index'

        FY = FY + [FaceY[i].index]

    setFY = it.faces[FY[0]:FY[0] + 1]

    for i in range(1, len(FY)):
        setFY += it.faces[FY[i]:FY[i] + 1]

    ## rt.Set(faces=setFX0, name='FX0')

    rt.Surface(name='FY', side1Faces=setFY)

    rt.Set(name='FY', faces=setFY)

    for i in range(0, len(FaceX)):
        FX = FX + [FaceX[i].index]

    setFX = it.faces[FX[0]:FX[0] + 1]

    for i in range(0, len(FX)):
        setFX += it.faces[FX[i]:FX[i] + 1]

    rt.Surface(name='FX', side1Faces=setFX)

    rt.Set(name='FX', faces=setFX)

    for i in range(0, len(FaceZ)):
        FZ = FZ + [FaceZ[i].index]

    setFZ = it.faces[FZ[0]:FZ[0] + 1]

    for i in range(0, len(FZ)):
        setFZ += it.faces[FZ[i]:FZ[i] + 1]

    rt.Surface(name='FZ', side1Faces=setFZ)

    rt.Set(name='FZ', faces=setFZ)

    for i in range(0, len(FaceX0)):
        FX0 = FX0 + [FaceX0[i].index]

    setFX0 = it.faces[FX0[0]:FX0[0] + 1]

    for i in range(0, len(FX0)):
        setFX0 += it.faces[FX0[i]:FX0[i] + 1]

    rt.Set(name='FX0', faces=setFX0)

    for i in range(0, len(FaceY0)):
        FY0 = FY0 + [FaceY0[i].index]

    setFY0 = it.faces[FY0[0]:FY0[0] + 1]

    for i in range(0, len(FY0)):
        setFY0 += it.faces[FY0[i]:FY0[i] + 1]

    rt.Set(name='FY0', faces=setFY0)

    for i in range(0, len(FaceZ0)):
        FZ0 = FZ0 + [FaceZ0[i].index]

    setFZ0 = it.faces[FZ0[0]:FZ0[0] + 1]

    for i in range(0, len(FZ0)):
        setFZ0 += it.faces[FZ0[i]:FZ0[i] + 1]

    rt.Set(name='FZ0', faces=setFZ0)

    m.Coupling(controlPoint=

               rt.sets['RP2'], couplingType=KINEMATIC,

               influenceRadius=WHOLE_SURFACE, localCsys=None, name='Constraint-1',

               surface=rt.surfaces['FY'], u1=ON, u2=

               ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    m.YsymmBC(createStepName='Step-1', localCsys=None, name=

    'BC-1', region=rt.sets['FY0'])

    m.ZsymmBC(createStepName='Step-1', localCsys=None, name=

    'BC-2', region=rt.sets['FZ0'])

    m.XsymmBC(createStepName='Step-1', localCsys=None, name=

    'BC-3', region=rt.sets['FX0'])

    m.DisplacementBC(amplitude=UNSET, createStepName='Step-1',

                     distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=

                     'BC-4', region=rt.sets['RP2'], u1=0,

                     u2=dis_y, u3=0, ur1=0, ur2=0, ur3=0)


def abaqus_BC(offspring, m, rt, q):
    ##self-contact behavior
    m.ContactProperty('IntProp-1')
    m.interactionProperties['IntProp-1'].NormalBehavior(
        allowSeparation=ON, constraintEnforcementMethod=DEFAULT, pressureOverclosure=HARD)
    rt.Surface(name='Surf-1', side1Faces=
    rt.instances['Merge-1-1'].faces.getByBoundingBox(-0.5, -0.5, -0.5, unit_lx_total + 0.5, unit_ly_total + 0.5,
                                                     unit_lz_total + 0.5))
    m.SelfContactStd(contactTracking=ONE_CONFIG,
                     createStepName='Step-1', interactionProperty='IntProp-1', name='Int-1',
                     surface=rt.surfaces['Surf-1'], thickness=ON)
    ##Boundary conditions
    it = rt.instances['Merge-1-1']
    FaceX0 = []
    print('facex0=', FaceX0)
    FaceX = []
    print('facex=', FaceX)
    FaceY0 = []
    print('facey0=', FaceY0)
    FaceY = []
    print('facey=', FaceY)
    FaceZ0 = []
    print('facez0=', FaceZ0)
    FaceZ = []
    print('facez=', FaceZ)

    ## select all Face

    frt_end = [0, lx - 1]
    for i in frt_end:
        for j in range(ly):
            for k in range(lz):
                if offspring[q - 1][k][j][i] == 1:
                    if i == 0:
                        FaceX0 = FaceX0 + [it.faces.findAt((0, unit_l * j + unit_l_half, unit_l * k + unit_l_half), )]
                        print('facex0_added')
                        print('coordinate', i, j, k)
                    else:
                        FaceX = FaceX + [
                            it.faces.findAt((unit_lx_total, unit_l * j + unit_l_half, unit_l * k + unit_l_half), )]
                        print('facex_added')
                        print('coordinate', i, j, k)
    for j in frt_end:
        for i in range(lx):
            for k in range(lz):
                if offspring[q - 1][k][j][i] == 1:
                    if j == 0:
                        FaceY0 = FaceY0 + [it.faces.findAt((unit_l * i + unit_l_half, 0, unit_l * k + unit_l_half), )]
                        print('facey0_added')
                        print('coordinate', i, j, k)
                    else:
                        FaceY = FaceY + [
                            it.faces.findAt((unit_l * i + unit_l_half, unit_ly_total, unit_l * k + unit_l_half), )]
                        print('facey_added')
                        print('coordinate', i, j, k)

    for k in frt_end:
        for i in range(lx):
            for j in range(ly):
                if offspring[q - 1][k][j][i] == 1:
                    if k == 0:
                        FaceZ0 = FaceZ0 + [it.faces.findAt((unit_l * i + unit_l_half, unit_l * j + unit_l_half, 0), )]
                        print('facez0_added')
                        print('coordinate', i, j, k)
                    else:
                        FaceZ = FaceZ + [
                            it.faces.findAt((unit_l * i + unit_l_half, unit_l * j + unit_l_half, unit_lz_total), )]
                        print('facez_added')
                        print('coordinate', i, j, k)
    print('facex=', FaceX)
    print('facex0=', FaceX0)
    print('facey=', FaceY)
    print('facey0=', FaceY0)
    print('facez=', FaceZ)
    print('facez0=', FaceZ0)
    FX = []
    FX0 = []
    FY = []
    FY0 = []
    FZ = []
    FZ0 = []
    for i in range(0, len(FaceY)):  # AttributeError: 'NoneType' object has no attribute 'index'
        FY = FY + [FaceY[i].index]
    setFY = it.faces[FY[0]:FY[0] + 1]

    for i in range(1, len(FY)):
        setFY += it.faces[FY[i]:FY[i] + 1]
    rt.Surface(name='FY', side1Faces=setFY)
    rt.Set(name='FY', faces=setFY)

    for i in range(0, len(FaceX)):
        FX = FX + [FaceX[i].index]
    setFX = it.faces[FX[0]:FX[0] + 1]

    for i in range(0, len(FX)):
        setFX += it.faces[FX[i]:FX[i] + 1]

    rt.Surface(name='FX', side1Faces=setFX)

    rt.Set(name='FX', faces=setFX)

    for i in range(0, len(FaceZ)):
        FZ = FZ + [FaceZ[i].index]

    setFZ = it.faces[FZ[0]:FZ[0] + 1]

    for i in range(0, len(FZ)):
        setFZ += it.faces[FZ[i]:FZ[i] + 1]

    rt.Surface(name='FZ', side1Faces=setFZ)

    rt.Set(name='FZ', faces=setFZ)

    for i in range(0, len(FaceX0)):
        FX0 = FX0 + [FaceX0[i].index]

    setFX0 = it.faces[FX0[0]:FX0[0] + 1]

    for i in range(0, len(FX0)):
        setFX0 += it.faces[FX0[i]:FX0[i] + 1]

    rt.Set(name='FX0', faces=setFX0)

    for i in range(0, len(FaceY0)):
        FY0 = FY0 + [FaceY0[i].index]

    setFY0 = it.faces[FY0[0]:FY0[0] + 1]

    for i in range(0, len(FY0)):
        setFY0 += it.faces[FY0[i]:FY0[i] + 1]

    rt.Set(name='FY0', faces=setFY0)

    for i in range(0, len(FaceZ0)):
        FZ0 = FZ0 + [FaceZ0[i].index]

    setFZ0 = it.faces[FZ0[0]:FZ0[0] + 1]

    for i in range(0, len(FZ0)):
        setFZ0 += it.faces[FZ0[i]:FZ0[i] + 1]

    rt.Set(name='FZ0', faces=setFZ0)

    m.Coupling(controlPoint=

               rt.sets['RP2'], couplingType=KINEMATIC,

               influenceRadius=WHOLE_SURFACE, localCsys=None, name='Constraint-1',

               surface=rt.surfaces['FY'], u1=ON, u2=

               ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    m.YsymmBC(createStepName='Step-1', localCsys=None, name=

    'BC-1', region=rt.sets['FY0'])

    m.ZsymmBC(createStepName='Step-1', localCsys=None, name=

    'BC-2', region=rt.sets['FZ0'])

    m.XsymmBC(createStepName='Step-1', localCsys=None, name=

    'BC-3', region=rt.sets['FX0'])

    m.DisplacementBC(amplitude=UNSET, createStepName='Step-1',

                     distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=

                     'BC-4', region=rt.sets['RP2'], u1=0,

                     u2=dis_y, u3=0, ur1=0, ur2=0, ur3=0)


def abaqus_historyOutput(w, q):
    if mode == 'Gaussian':
        if model == 'original':
            f = openOdb(path='Job-{}-{}_original.odb'.format(w, q))
        if model == 'gaussian':
            f = openOdb(path='Job-{}-{}_gaussian.odb'.format(w, q))

    else:
        f = openOdb(path='Job-{}-{}.odb'.format(w, q))
    region = f.steps['Step-1'].historyRegions['Node ASSEMBLY.1'].historyOutputs
    u1 = np.array(region['U1'].data)  # type :tuple
    u2 = np.array(region['U2'].data)
    u3 = np.array(region['U3'].data)
    rf1 = np.array(region['RF1'].data)
    rf2 = np.array(region['RF2'].data)
    rf3 = np.array(region['RF3'].data)

    if mode == 'Random':
        w = 0

    file_names = ('U1_HistoryOutput', 'U2_HistoryOutput', 'U3_HistoryOutput',
                  'RF1_HistoryOutput', 'RF2_HistoryOutput', 'RF3_HistoryOutput')
    arrs = (u1, u2, u3, rf1, rf2, rf3)
    for arr in arrs:
        print(arr.shape)

    for file_name, arr in zip(file_names, arrs):
        file_name = '{}_{}'.format(file_name, w)
        if os.path.isfile(file_name):
            with open(file_name, mode='rb') as f_history_read:
                history_output_read = pickle.load(f_history_read)
            with open(file_name, mode='wb') as f_history_write:
                history_output_read.update({q: arr.transpose()})
                pickle.dump(history_output_read, f_history_write)
        else:
            with open(file_name, mode='wb') as f_history_write:
                pickle.dump({q: arr.transpose()}, f_history_write)


def abaqus_fieldOutput(w, q):
    if mode == 'Gaussian':
        if model == 'original':
            f = openOdb(path='Job-{}-{}_original.odb'.format(w, q))
        if model == 'gaussian':
            f = openOdb(path='Job-{}-{}_gaussian.odb'.format(w, q))

    else:
        f = openOdb(path='Job-{}-{}.odb'.format(w, q))
    Disp = f.steps['Step-1'].frames[-1].fieldOutputs['U']
    ReForce = f.steps['Step-1'].frames[-1].fieldOutputs['RF']
    Ro = f.steps['Step-1'].frames[-1].fieldOutputs['UR']
    Stress = f.steps['Step-1'].frames[-1].fieldOutputs['S'].values
    nodesetFX = f.rootAssembly.nodeSets['FX']
    nodesetFY = f.rootAssembly.nodeSets['FY']
    nodesetFZ = f.rootAssembly.nodeSets['FZ']
    nodesetRP = f.rootAssembly.nodeSets['RP2']
    subsFX = Disp.getSubset(region=nodesetFX)
    subsFY = Disp.getSubset(region=nodesetFY)
    subsFZ = Disp.getSubset(region=nodesetFZ)
    RF_RP = ReForce.getSubset(region=nodesetRP).values[0].data
    Ro_RP = Ro.getSubset(region=nodesetRP).values[0].data
    disX = []
    disY = []
    disZ = []
    mises = []

    for i in subsFX.values:
        disX.append(i.data[0])

    for i in subsFY.values:
        disY.append(i.data[1])

    for i in subsFZ.values:
        disZ.append(i.data[2])

    for i in Stress:
        mises.append(i.mises)

    max_mises = max(mises)
    min_mises = min(mises)
    avg_mises = 0
    dis11 = 0
    dis22 = 0
    dis33 = 0

    for i in range(len(mises)):
        avg_mises += mises[i] / len(mises)
    for i in range(len(disX)):
        dis11 += disX[i] / len(disX)
    for i in range(len(disY)):
        dis22 += disY[i] / len(disY)
    for i in range(len(disZ)):
        dis33 += disZ[i] / len(disZ)

    dis = []
    dis.append(dis11)
    dis.append(dis22)
    dis.append(dis33)

    misess = []
    misess.append(max_mises)
    misess.append(min_mises)
    misess.append(avg_mises)
    try:
        eng_const = np.reshape(np.concatenate((dis, RF_RP, Ro_RP, misess)), newshape=(1, 12))
    except:
        print('error')

    if mode == 'Random':
        file_name = 'Output_parent_{}.csv'.format(1)
        array_to_csv(path=file_name, arr=eng_const[0, :], dtype=np.float32, mode='a')

    else:
        file_name = 'Output_offspring_{}.csv'.format(w)
        array_to_csv(path=file_name, arr=eng_const[0, :], dtype=np.float32, mode='a')

    if mode == 'Gaussian':
        if model == 'original':
            mdb.Model(modelType=STANDARD_EXPLICIT, name='Model-%d_original' % (q + 1))
            del mdb.models['Model-%d_original' % (q)]
            del mdb.jobs['Job-%d-%d_original' % (w, q)]

        if model == 'gaussian':
            mdb.Model(modelType=STANDARD_EXPLICIT, name='Model-%d_gaussian' % (q + 1))
            del mdb.models['Model-%d_gaussian' % (q)]
            del mdb.jobs['Job-%d-%d_gaussian' % (w, q)]

    else:
        mdb.Model(modelType=STANDARD_EXPLICIT, name='Model-%d' % (q + 1))
        del mdb.models['Model-%d' % (q)]
        del mdb.jobs['Job-%d-%d' % (w, q)]


def abaqus_jobsubmit(w, q):
    if mode == 'GA' or mode == 'Random' or mode == 'None':
        mdb.jobs['Job-%d-%d' % (w, q)].writeInput()

        mdb.jobs['Job-%d-%d' % (w, q)].submit(consistencyChecking=OFF)

        mdb.jobs['Job-%d-%d' % (w, q)].waitForCompletion()

    if mode == 'Gaussian':

        if model == 'original':
            mdb.jobs['Job-%d-%d_original' % (w, q)].writeInput()

            mdb.jobs['Job-%d-%d_original' % (w, q)].submit(consistencyChecking=OFF)

            mdb.jobs['Job-%d-%d_original' % (w, q)].waitForCompletion()

        if model == 'gaussian':
            mdb.jobs['Job-%d-%d_gaussian' % (w, q)].writeInput()

            mdb.jobs['Job-%d-%d_gaussian' % (w, q)].submit(consistencyChecking=OFF)

            mdb.jobs['Job-%d-%d_gaussian' % (w, q)].waitForCompletion()


def control_abaqus(w, offspring, frame=None, restart=False):
    if restart:
        save_log('Restarting Job{}-{}.odb'.format(w, restart_pop), frame=frame)
        mdb.Model(modelType=STANDARD_EXPLICIT, name='Model-%d' % restart_pop)
        del mdb.models['Model-1']
        for q in range(restart_pop, end_pop + 1):
            m = mdb.models['Model-%d' % q]
            rt = m.rootAssembly
            abaqus_cad(offspring, m, rt, q)
            abaqus_merge4(5, m, rt)
            # save_log('merge complete', frame=frame)
            abaqus_mesh(m)
            # save_log('mesh complete', frame=frame)
            abaqus_material_anisotropic(offspring, 1500, 1200, 1500, 0.35, 0.35, 0.35, 450, 550, 450, m, q)
            # save_log('material complete', frame=frame)
            abaqus_step(m, rt)
            # save_log('step complete', frame=frame)
            abaqus_BC(offspring, m, rt, q)
            # save_log('BC complete', frame=frame)
            abaqus_jobsetting(w, q)
            # save_log('job setting complete', frame=frame)
            abaqus_jobsubmit(w, q)
            # save_log('job submit complete', frame=frame)
            abaqus_fieldOutput(w, q)
            # save_log('field output complete', frame=frame)
            # save_log('[{}][Gen{} offspring{}] Field output export complete!'.format(now_s(), w, q))
            abaqus_historyOutput(w, q)  # correction: input () >> (w,q)
            # save_log('history output complete', frame=frame)
            save_log('[{}][Gen{}] {}/{} ({:.2f}%) Complete'.format(now_s(), w, q, end_pop, float(q)/float(end_pop) * 100),
                     frame=frame)
        mdb.Model(modelType=STANDARD_EXPLICIT, name='Model-1')
        del mdb.models['Model-%d' % (end_pop + 1)]
        save_log('[{}] ========== All Generation{} work done! =========='.format(now_s(), w), frame=frame)
        os.remove('./args')
    else:
        for q in range(ini_pop, end_pop + 1):
            m = mdb.models['Model-%d' % q]
            rt = m.rootAssembly
            abaqus_cad(offspring, m, rt, q)
            abaqus_merge4(5, m, rt)
            abaqus_mesh(m)
            abaqus_material_anisotropic(offspring, 1500, 1200, 1500, 0.35, 0.35, 0.35, 450, 550, 450, m, q)
            abaqus_step(m, rt)
            abaqus_BC(offspring, m, rt, q)
            abaqus_jobsetting(w, q)
            abaqus_jobsubmit(w, q)
            abaqus_fieldOutput(w, q)
            # save_log('[{}][Gen{} offspring{}] Field output export complete!'.format(now_s(), w, q))
            abaqus_historyOutput(w, q)  # correction: input () >> (w,q)
            save_log('[{}][Gen{}] {}/{} ({:.2f}%) Complete'.format(now_s(), w, q, end_pop, float(q)/float(end_pop) * 100),
                     frame=frame)
        mdb.Model(modelType=STANDARD_EXPLICIT, name='Model-1')
        del mdb.models['Model-%d' % (end_pop + 1)]
        save_log('[{}] ========== All Generation{} work done! =========='.format(now_s(), w), frame=frame)
        os.remove('./args')


# The main part begins...
frame = open_log_window()
while True:
    print('[{}] ..... scanning for args or args_end .....'.format(now_s()))
    if os.path.isfile('./args'):
        # save_log('[{}] args found'.format(now_s()))
        with open('./args', mode='rb') as f_args:
            args = pickle.load(f_args)
        restart = args['restart']
        w = args['w']
        offspring = args['offspring']
        control_abaqus(w=w, offspring=offspring, frame=frame, restart=restart)
    elif os.path.isfile('./args_end'):
        os.remove('./args_end')
        break
    sleep(1)
save_log('ABAQUS SESSION DONE!', frame=frame)
