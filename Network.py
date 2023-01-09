import socket
import threading
import json
from datetime import datetime
from time import sleep
import pickle
import struct
from sys import version_info
from queue import Queue


class Server:
    def __init__(self, host, port, option, run_nonblocking):
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
            self._server_th = threading.Thread(target=self.run)
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

    def send(self, client_socket, data):
        if self.option == 'pickle':
            serialized_data = pickle.dumps(data, protocol=2)
        else:
            serialized_data = b''
        client_socket.sendall(struct.pack(self._header_format, len(serialized_data)))
        client_socket.sendall(serialized_data)
        print('[{}] A data sent'.format(datetime.now()))

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
            serialized_data = b''
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


if __name__ == '__main__':
    open_server = True
    if open_server:  # Creating server
        server = Server(host='', port=9999, option='pickle', run_nonblocking=True)
        print('server created')
        while True:
            datum = server.recv()
            print(f'- A data from queue: {datum}\n')

    else:  # Creating client
        import numpy as np
        print('client created')
        gateway = '115.145.177.1'
        external_ip_address = '115.145.177.126'
        client = Client(host=external_ip_address, port=9999, option='pickle', connect=True)
        msg_sent_count = 0
        while msg_sent_count < 100000000000000000:
            client.send(f'Random number: {np.random.random(3)}')
            sleep(0.5)
            msg_sent_count += 1
        client.client_socket.close()
    print('sleeping...')
    sleep(1000)
