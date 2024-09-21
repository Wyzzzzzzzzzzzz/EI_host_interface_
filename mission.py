import socket
import struct
import json


        
        
        
def tcp_navigation (location = 0):
    print("tcp发送move")
    buffer_data = bytearray(6)
    buffer_data[0] = 0xFE
    buffer_data[1] = 0x03
    buffer_data[2] = 0x03
    buffer_data[3] = 0x01
    buffer_data[4] = 0x00
    buffer_data[5] = 0x00


    buffer_data[4] = location
    i = 1
    while i < 5:
        buffer_data[5] = (buffer_data[5] + buffer_data[i])%256
        i+=1
    tcp.send_data(buffer_data)  
    is_move_finished = 0
    while(is_move_finished == 0):
        buffer_received = tcp.receive_data()
        if not buffer_received:
            continue
        if buffer_received[0] == 0xFE and buffer_received[1] == 0x03 and buffer_received[2] == 0x03 and buffer_received[3] == 0x01:
            i = 1
            numsum = 0
            while i < 5:
                numsum = (numsum + buffer_received[i])%256
                i+=1
            if numsum == buffer_received[5]:
                is_move_finished = buffer_received[4]
                print("移动成功，到达目标地点")
