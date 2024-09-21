import socket
import struct
import json
import threading
from PIL import Image
from PIL import ImageFont, ImageDraw
import cv2
from ultralytics import YOLOWorld
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import torch
import sys
import argparse
import os
import base64
model = YOLO('yolov8n-pose.pt')
def video_receiver(ip, port):
    # 创建一个UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 绑定到特定的IP和端口
    sock.bind((ip, port))
    sock.settimeout(3)
    print("接收器启动，等待数据...")

    while True:
        # 接收数据
        try:
            data, addr = sock.recvfrom(30720)  # 1024表示缓冲区大小，可以根据需要调整
        except socket.timeout:
            x=1
            y=1
            frame_header = 0xff
            data_format = '!BHH'  # '!B' 表示 big-endian unsigned char, 'HH' 表示两个 big-endian unsigned shorts
            byte_data = struct.pack(data_format, frame_header, x, y)
            sock.sendto(byte_data, ("10.78.76.47",1235))
        else:
            # 将字节数据转换为NumPy数组
            color_image = np.frombuffer(data, dtype=np.uint8)
            
            # 解码图像
            color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
            
            results = model.track(color_image,classes=[0],conf=0.5,show=True)
            annotated_frame = results[0].plot()
            
            if(len(results[0].keypoints.xy[0]) != 0) :
                if(results[0].keypoints.xy[0][11][0] != 0):
                    x = results[0].keypoints.xy[0][11][0]#只针对识别到的第一个对象results[0].keypoints.xy
                    y = results[0].keypoints.xy[0][11][1]
                    x = int(x)
                    y = int(y)
            else:
                x = 1
                y = 1
            frame_header = 0xff
            data_format = '!BHH'  # '!B' 表示 big-endian unsigned char, 'HH' 表示两个 big-endian unsigned shorts
            byte_data = struct.pack(data_format, frame_header, x, y)
            print(x)
            print(y)
            #_, buffer = cv2.imencode('.jpeg', annotated_frame,[int(cv2.IMWRITE_JPEG_QUALITY), 70])
            #byte_data = buffer.tobytes()
            sock.sendto(byte_data, ("10.78.76.47",1235))
            print("up_send")
            # 显示图像
            cv2.imshow('receive', annotated_frame)
            # 按'q'退出
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
    # 释放资源
    sock.close()
    cv2.destroyAllWindows()
    
    
    
video_receiver("10.78.76.37",1235)



