import socket
import threading
import json
import os
import pickle
import struct
import multiprocessing as mp
import inspect
from multiprocessing import connection
from datetime import datetime
from time import sleep
from sys import version_info
from typing import Tuple
try:
    from Queue import Queue
except ImportError:
    from queue import Queue


class Server:
    def __init__(self, host, port, option, run_nonblocking):
        parent_frame = inspect.stack()[1][0]
        parent_frame_name = inspect.getmodule(parent_frame).__name__
        if parent_frame_name != '__main__':
            raise SystemExit(f'[Error] Server is not created within conditional block if __name__=="__main__".'
                             f' Use conditional block!')
        self.host = host
        self.port = port
        self.option = option
        self.q = Queue()
        self.connected_clients = list()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self._default_packet_size = 1024
        self._header_format = '>I'
        self._header_bytes = 4
        if run_nonblocking:
            self._server_th = threading.Thread(target=self.run, daemon=True)
            self._server_th.start()

    def run(self):
        while True:
            try:
                print('[{}] Waiting for a client...'.format(datetime.now()))
                client_socket, client_addr = self.server_socket.accept()
                if version_info.major >= 3:
                    new_th = threading.Thread(target=self._thread_recv, args=(client_socket, client_addr, self.option),
                                              daemon=True)
                else:
                    new_th = threading.Thread(target=self._thread_recv, args=(client_socket, client_addr, self.option))
                    new_th.setDaemon(True)
                new_th.start()
                self.connected_clients.append(client_socket)
            except Exception as e:
                print('<!> Server error:', e)
                break
        self.server_socket.close()

    def send(self, client_socket: socket.socket, data: any) -> bool:
        if self.option == 'pickle':
            serialized_data = pickle.dumps(data, protocol=2)
        else:
            serialized_data = json.dumps(data).encode()
        try:
            print('Sending packets: ', serialized_data)
            client_socket.sendall(struct.pack(self._header_format, len(serialized_data)))
            client_socket.sendall(serialized_data)
            return True
        except ConnectionError as e:
            print(e)
            return False

    def recv(self):
        return self.q.get()

    def close(self):
        self.server_socket.close()

    def _thread_recv(self, client_socket, client_addr, option):
        print('[{}] {}:{} has joined!'.format(datetime.now(), client_addr[0], client_addr[1]))
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
                print('Sending packets: ', serialized_data)
                self.client_socket.sendall(struct.pack(self._header_format, len(serialized_data)))
                self.client_socket.sendall(serialized_data)
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


def make_and_start_process(target: any, duplex: bool = True,
                           daemon: bool = True) -> Tuple[mp.Process, connection.Connection, connection.Connection]:
    """
    Make GUI process and return a process and two Pipe connections between main process and GUI process.
    :param target: The GUI class to run as another process.
    :param duplex: If True, both receiving and sending data between main process and GUI process will be allowed.
    Otherwise, conn_1 is only allowed for receiving data and conn_2 is only allowed for sending data.
    :param daemon: If True, GUI process will be terminated when main process is terminated.
    Otherwise, GUI process will be orphan process.
    :return: A running process, Pipe connections of main process, GUI process, respectively.
    """
    conn_1, conn_2 = mp.Pipe(duplex=duplex)
    process = mp.Process(target=target, args=(conn_2,), daemon=daemon)
    process.start()
    return process, conn_1, conn_2


def start_abaqus_cae() -> mp.Process:
    """
    Open an abaqus CAE process
    :param option: 'noGUI' for abaqus non-gui mode, 'script' for abaqus gui mode.
    :return: Abaqus process.
    """
    print(f"========== Opening ABAQUS on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}! ==========")
    script_path = os.path.join(os.path.dirname(__file__), 'AbaqusScripts.py')
    process = mp.Process(target=os.system, args=(f'abaqus cae script={script_path}',), daemon=True)
    process.start()  # Start abaqus
    return process


def request_abaqus(dict_data: dict, server: Server, conn_to_gui: connection.Connection) -> None:
    """
    Send Json data to ABAQUS
    :param conn_to_gui: Pipe connection to GUI
    :param dict_data: Dictionary data to send to ABAQUS
    :param server: A server to ABAQUS
    :return: Nothing
    """
    while len(server.connected_clients) == 0:
        print('Waiting for ABAQUS socket connection ...')
        sleep(1.0)
    is_data_sent = False
    while not is_data_sent:
        is_data_sent = server.send(client_socket=server.connected_clients[-1], data=dict_data)
    print('Waiting for message from ABAQUS ...')
    while True:
        json_data_from_client = server.recv()
        conn_to_gui.send({'log_message': json_data_from_client['log_message']})
        if json_data_from_client['end_generation']:
            break
    print(f"========== An evolution on ABAQUS is done on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}! ==========")


if __name__ == '__main__':
    open_server = False
    if open_server:  # Creating server
        my_server = Server(host='', port=12345, option='json', run_nonblocking=True)
        print('server created')
        while len(my_server.connected_clients) == 0:
            sleep(1)
        while True:
            data_to_send = {'a': 1, 'b': 2.0, 'c': 'hello', 'd': True, 'e': [0, 1, 2, 3]}
            json_data = json.dumps(data_to_send)
            my_server.send(client_socket=my_server.connected_clients[-1], data=data_to_send)
            print('sending: ', json_data)
            sleep(5)

    else:  # Creating client
        my_client = Client(host='localhost', port=12345, option='json', connect=True)
        while True:
            print('Received: ', my_client.recv())
