import os
from auxeticmop import find_job_location_from_offspring, Parameters
from dataclasses import asdict

#
# def _get_numbers(p):
#     return sorted(
#         [int(re.compile(r'\d+').search(s).group()) for s in [f for f in os.listdir() if re.compile(p).match(f)]])
#
#
# def _load_whole_files(header, nd, ad):
#     if 'topo' in header:
#         dtype = int
#     else:
#         dtype = float
#     nd.update({header: _get_numbers(rf'{header}_\d+\.csv')})
#     ad.update({header: np.expand_dims(np.genfromtxt(f'{header}_{nd[header][0]}.csv',
#                                                     dtype=dtype, delimiter=','), axis=0)})
#     for file_idx, num in enumerate(nd[header]):
#         if file_idx == 0:
#             continue
#         ad[header] = np.vstack((ad[header],
#                                 np.expand_dims(np.genfromtxt(f'{header}_{num}.csv',
#                                                              dtype=dtype, delimiter=','), axis=0)))
#     return ad
#
#
# def result_arr_to_dict(arrs):
#     result_dict = dict()
#     for entity_num, arr in enumerate(arrs, start=1):
#         dis_x, dis_y, dis_z = arr[0], arr[1], arr[2]
#         rf_x, rf_y, rf_z = arr[3], arr[4], arr[5]
#         ro_x, ro_y, ro_z = arr[6], arr[7], arr[8]
#         mss_max, mss_min, mss_avg = arr[9], arr[10], arr[11]
#
#         exported_field_outputs_format = {
#             'displacement': {'xMax': np.array([dis_x, 0, 0]), 'yMax': np.array([0, dis_y, 0]),
#                              'zMax': np.array([0, 0, dis_z])},
#             'rotation': np.array([ro_x, ro_y, ro_z]),
#             'reaction_force': np.array([rf_x, rf_y, rf_z]),
#             'mises_stress': {'max': mss_max, 'min': mss_min, 'average': mss_avg}
#         }
#         result_dict.update({entity_num: exported_field_outputs_format})
#     return result_dict
#
#
# def csv_to_pickle():
#     filename_headers = ('topo_parent', 'topo_offspring', 'Output_parent', 'Output_offspring')
#     num_dict, arr_dict = dict(), dict()
#     threads = [Thread(target=_load_whole_files, args=(header, num_dict, arr_dict), daemon=True)
#                for header in filename_headers]
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()
#     del threads
#     for arr_idx, num in enumerate(num_dict['Output_parent']):
#         result_arrs = arr_dict['Output_parent'][arr_idx]
#         result_dict = result_arr_to_dict(result_arrs)
#         asyncio.run(pickle_io(f'FieldOutput_{num}', mode='w', to_dump=result_dict))
#     for arr_idx, num in enumerate(num_dict['Output_offspring']):
#         result_arrs = arr_dict['Output_offspring'][arr_idx]
#         result_dict = result_arr_to_dict(result_arrs)
#         asyncio.run(pickle_io(f'FieldOutput_offspring_{num}', mode='w', to_dump=result_dict))
#     for arr_idx, num in enumerate(num_dict['topo_parent']):
#         topo_parent = arr_dict['topo_parent'][arr_idx]
#         topo_dict = {'parent': topo_parent}
#         asyncio.run(pickle_io(f'Topologies_{num}', mode='a', to_dump=topo_dict))
#     for arr_idx, num in enumerate(num_dict['topo_offspring']):
#         topo_offspring = arr_dict['topo_offspring'][arr_idx]
#         topo_dict = {'offspring': topo_offspring}
#         asyncio.run(pickle_io(f'Topologies_{num}', mode='a', to_dump=topo_dict))


if __name__ == '__main__':
    path = r'C:\pythoncode\AuxeticMOP\abaqus data'
    parameters = Parameters()
    parameters.post_initialize()
    os.chdir(path)
    find_job_location_from_offspring(params_dict=asdict(parameters))

