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

from scripts.utils import print_with_color
from utils_agent import *           # 智能体Agent编排
from utils_asr import *             # 录音+语音识别
from utils_tts import *             # 语音合成模块
font = ImageFont.truetype('asset/SimHei.ttf', 26)
#play_wav('asset/welcome.wav')
#response = "快来人快人，老人摔倒了，请根据手机定位呼叫救援"
#tts(response)                     # 语音合成，导出wav音频文件
#play_wav('temp/tts.wav')          # 播放语音合成音频文件
class TCPClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connect()

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.server_port))

    def send_data(self, data):
        # Example: Assuming data is a dictionary that you want to serialize to bytes
        bdata = bytes(data)
        
        self.socket.sendall(bdata)

    def receive_data(self, buffer_size=4096):
        received_data = self.socket.recv(buffer_size)
        return received_data

    def close(self):
        if self.socket:
            self.socket.close()

    def serialize_data(self, data):
        # Implement your serialization method here
        # Example: Convert a dictionary to JSON and encode as bytes
        serialized_data = json.dumps(data,cls=MyEncoder,indent=4).encode('utf-8')
        return serialized_data

    def deserialize_data(self, data):
        # Implement your deserialization method here
        # Example: Decode bytes to JSON and parse into a dictionary
        deserialized_data = json.loads(data.decode('utf-8'))
        return deserialized_data

print("6666")
#初始化tcp、摄像头、yolo-world
#tcp = TCPClient("10.78.76.47", 12345)
#tcp = TCPClient("10.147.15.168", 12345)
#pipeline = rs.pipeline()
#cfg = rs.config()
#cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#align_to = rs.stream.color# 设定需要对齐的方式（这里是深度对齐彩色，彩色图不变，深度图变换）
#alignedFs = rs.align(align_to)
#profile = pipeline.start(cfg)
model = YOLOWorld('yolov8l-world.pt')
model_pose = YOLO('yolov8l-pose.pt')
print("tcp连接成功")

def video_sender(ip, port):
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        fs = pipeline.wait_for_frames()
        aligned_frames = alignedFs.process(fs)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        _, buffer = cv2.imencode('.jpeg', color_image,[int(cv2.IMWRITE_JPEG_QUALITY), 50])
        byte_data = buffer.tobytes()
        sock.sendto(byte_data, (ip, port))

def tcp_navigation (location = 0):
    print("tcp发送move")
    buffer_data = bytearray(6)
    buffer_data[0] = 0xFE
    buffer_data[1] = 0x06
    buffer_data[2] = 0x06
    buffer_data[3] = 0x01
    buffer_data[4] = 0x00
    buffer_data[5] = 0x00


    buffer_data[4] = location
    print("发送数据"+buffer_data)
    i = 1
    while i < 5:
        buffer_data[5] = (buffer_data[5] + buffer_data[i])%256
        i+=1
    tcp.send_data(buffer_data)  
    is_move_finished = 0
    while(is_move_finished == 0):
        buffer_received = tcp.receive_data()
        print("接受到了数据")
        if buffer_received[0] == 0xFE and buffer_received[1] == 0x06 and buffer_received[2] == 0x00 and buffer_received[3] == 0x01:
            
            i = 1
            numsum = 0
            while i < 5:
                numsum = (numsum + buffer_received[i])%256
                i+=1
            print("解包成功")
            if numsum == buffer_received[5]:
                is_move_finished = buffer_received[4]
                print("移动成功，到达目标地点")
                
def tcp_test():
    buffer_data = bytearray(17)
    buffer_data[0] = 0xFE
    buffer_data[1] = 0x05
    buffer_data[2] = 0x05
    buffer_data[3] = 0x0c
    uint_x = -100
    buffer_data[4] = uint_x%256
    buffer_data[5] = (uint_x>>8)%256
    buffer_data[6] = (uint_x>>16)%256
    buffer_data[7] = (uint_x>>24)%256
    uint_y = 10
    uint_y += 10000
    buffer_data[8] = uint_y%256
    buffer_data[9] = (uint_y>>8)%256
    buffer_data[10] = (uint_y>>16)%256
    buffer_data[11] = (uint_y>>24)%256
    uint_z = 100
    uint_z += 10000
    buffer_data[12] = uint_z%256
    buffer_data[13] = (uint_z>>8)%256
    buffer_data[14] = (uint_z>>16)%256
    buffer_data[15] = (uint_z>>24)%256
    buffer_data[16] = 0x00
    i = 1
    while i < 16:
        buffer_data[16] = (buffer_data[16] + buffer_data[i])%256
        i+=1
    tcp.send_data(buffer_data)
    cv2.waitKey(100000)


#trace发送的是速度数据，
def tcp_trace(obj = 'person'):

    model.set_classes([obj])
    print("trace_thread")
    targetpoint = [0, 0, 100]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("10.78.76.37", 1236))

    is_trace_finished = 0
    while(is_trace_finished == 0):
        fs = pipeline.wait_for_frames()
        aligned_frames = alignedFs.process(fs)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        #最终图像
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image_flipped_both = np.flip(color_image, (0, 1))

        _, buffer = cv2.imencode('.jpeg', color_image_flipped_both,[int(cv2.IMWRITE_JPEG_QUALITY), 50])
        byte_data = buffer.tobytes()
        sock.sendto(byte_data, ("10.78.76.37",1235))
        print("down_send")
        print(len(byte_data))
        #results = model.predict(color_image_flipped_both)
        data, addr = sock.recvfrom(30720)  # 1024表示缓冲区大小，可以根据需要调整
        if data:
            print("down_receive")
            #将字节数据转换为NumPy数组
            color_image_np = np.frombuffer(data, dtype=np.uint8)
           
            # 解码图像
            color_image_np = cv2.imdecode(color_image_np, cv2.IMREAD_COLOR)



        buffer_data = bytearray(8)
        buffer_data[0] = 0xFE
        buffer_data[1] = 0x05
        buffer_data[2] = 0x05
        buffer_data[3] = 0x03
        buffer_data[4] = 0x00
        buffer_data[5] = 0x00
        buffer_data[6] = 0x00
        buffer_data[7] = 0x0d
        #if(len(results[0].boxes.xywh) != 0):
        if 0:
            #x = results[0].boxes.xywh[0][0]#只针对识别到的第一个对象
            #y = results[0].boxes.xywh[0][1]
            #x = 640 - x
            #y = 480 - y
            x = 100
            y=100

            depth_pixel = [x, y]
            print(depth_pixel)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics# 获取相机参数
            dis = depth_frame.get_distance(int(x), int(y))
            camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis) 
            #摄像头正前方是dz 越远离越大
            #摄像头左右是dx 越向右越大（反转了，所以向左为正）
            #摄像头上下是dy 越向下越大（反转了，所以向上为正）
            dx = -camera_coordinate[0]*100
            dy = -camera_coordinate[1]*100#符号为了调整成方向，调整之后恢复原来的描述
            dz = camera_coordinate[2]*100
            
            mystring = str(round(dx, 2))+" "+str(round(dy, 2))+" "+str(round(dz, 2))
            #cv2.putText(annotated_frame, mystring, (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
            print(mystring)
            
            buffer_data[0] = 0xFE
            buffer_data[1] = 0x05
            buffer_data[2] = 0x05
            buffer_data[3] = 0x03
            if(dx > 5): 
                #buffer_data[4] = 0x01#向右
                buffer_data[4] = 0x02#向左
            elif(dx < -5): 
                #buffer_data[4] = 0x02#向左
                buffer_data[4] = 0x01#向右
            else: 
                buffer_data[4] = 0x00
            if(dz > 105): 
                buffer_data[5] = 0x01#向前
                print("向前")
            elif(dz < 95): 
                buffer_data[5] = 0x02#向后
                print('向后')
            else: 
                buffer_data[5] = 0x00
            if(dy > 5): 
                buffer_data[6] = 0x01#向下
            elif(dy < -5): 
                buffer_data[6] = 0x02#向上
            else: 
                buffer_data[6] = 0x00
            buffer_data[7] = 0x00
            i = 1
            while i < 7:
                buffer_data[7] = (buffer_data[7] + buffer_data[i])%256
                i+=1
            
        #tcp.send_data(buffer_data)
        cv2.imshow("image", color_image_np)
        if cv2.waitKey(100)&0xff==ord('q'):
            break

def tcp_catch(obj = 'bottle'):
    model.set_classes([obj])
    print("tcp发送catch")
    while(1):
        fs = pipeline.wait_for_frames()
        
        aligned_frames = alignedFs.process(fs)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        #最终图像
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image_flipped_both = np.flip(color_image, (0, 1))
        results = model.predict(color_image_flipped_both)
        annotated_frame = results[0].plot()
        if(len(results[0].boxes.xywh) != 0):
            x = results[0].boxes.xywh[0][0]#只针对识别到的第一个对象
            y = results[0].boxes.xywh[0][1]
            x = 640 - x
            y = 480 - y
            depth_pixel = [x, y]
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics# 获取相机参数
            dis = depth_frame.get_distance(int(x), int(y))
            camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis) 
            #摄像头正前方是dz 越远离越大
            #摄像头左右是dx 越向右越大(反转了，所以向左为正)
            #摄像头上下是dy 越向下越大（反转了，所以向左为正）
            dx = -camera_coordinate[0]*100
            dy = -camera_coordinate[1]*100#符号为了调整成方向，调整之后恢复原来的描述
            dz = camera_coordinate[2]*100
            mystring = str(round(dx, 2))+" "+str(round(dy, 2))+" "+str(round(dz, 2))
            cv2.putText(annotated_frame, mystring, (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
            print(mystring)
            buffer_data = bytearray(17)
            buffer_data[0] = 0xFE
            buffer_data[1] = 0x04
            buffer_data[2] = 0x04
            buffer_data[3] = 0x0c
            uint_x = int(dx)
            uint_x += 10000
            print(uint_x)
            buffer_data[4] = uint_x%256
            buffer_data[5] = (uint_x>>8)%256
            buffer_data[6] = (uint_x>>16)%256
            buffer_data[7] = (uint_x>>24)%256
            uint_y = int(dy)
            uint_y += 10000
            buffer_data[8] = uint_y%256
            buffer_data[9] = (uint_y>>8)%256
            buffer_data[10] = (uint_y>>16)%256
            buffer_data[11] = (uint_y>>24)%256
            uint_z = int(dz)
            uint_z += 10000
            buffer_data[12] = uint_z%256
            buffer_data[13] = (uint_z>>8)%256
            buffer_data[14] = (uint_z>>16)%256
            buffer_data[15] = (uint_z>>24)%256
            buffer_data[16] = 0x00
            i = 1
            while i < 16:
                buffer_data[16] = (buffer_data[16] + buffer_data[i])%256
                i+=1
            #tcp.send_data(buffer_data)
        cv2.imshow("image", annotated_frame)
        cv2.waitKey(50)
            
    is_catch_finished = 0
    while(is_catch_finished == 0):
        buffer_received = tcp.receive_data()
        print("接受到了数据")
        if buffer_received[0] == 0xFE and buffer_received[1] == 0x04 and buffer_received[2] == 0x00 and buffer_received[3] == 0x01:
            
            i = 1
            numsum = 0
            while i < 5:
                numsum = (numsum + buffer_received[i])%256
                i+=1
            print("解包成功")
            if numsum == buffer_received[5]:
                is_catch_finished = buffer_received[4]
                print("抓取成功")
                
                
def yi_vision_api(prompt='帮我把红色方块放在钢笔上', img_path='temp/vl_now.jpg', flag=1):
    SYSTEM_PROMPT1 = '''
    我即将说一句给机械臂的指令，你帮我从这句话中提取出物体并从这张图中分别找到物体左上角和右下角的像素坐标，输出json数据结构。

    例如，如果我的指令是：红色方块。
    你输出这样的格式：
    {
     "start":"红色方块",
     "start_xyxy":[[102,505],[324,860]],
    }
    例如，如果我的指令是：绿色瓶子。
    你输出这样的格式：
    {
     "start":"绿色瓶子",
     "start_xyxy":[[99,560],[234,608]],
    }

    
    只回复json本身即可，不要回复其它内容
    注意
    只有[[300,150],[476,310]]的格式是正确的
    一个[]中只能有两个数据例如[12,59]
    我现在的指令是：
    '''
    
    SYSTEM_PROMPT0 = '''
    我将会向你询问画面中物体的情况，你需要实际情况回答
    例子：
    我说：描述一下你面前的场景。你回答："我看到了一支笔、一个水瓶和一个摄像头。"
    我说：告诉我前面的黑色长方形物品是什么。你回答："那是显示器，用于显示电脑主机的画面"
    请仿照例子，用不超过三十个字的回答用中文简单总结回答，
    现在 我说
    '''
    SYSTEM_PROMPT=""
    if flag==1:
        SYSTEM_PROMPT=SYSTEM_PROMPT1
    else:
        SYSTEM_PORMPT=SYSTEM_PROMPT0
    API_BASE = "https://api.lingyiwanwu.com/v1"
    API_KEY = "bd3dd4e23ab2402aa88eb12692ea64be"
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE
    )
    
    # 编码为base64数据
    with open(img_path, 'rb') as image_file:
        image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')

    # 向大模型发起请求
    completion = client.chat.completions.create(
      model="yi-vision",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": SYSTEM_PROMPT + prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": image
              }
            }
          ]
        },
      ]
    )
    
    # 解析大模型返回结果
    print(completion.choices[0].message.content.strip())
    if(flag == 0):result = completion.choices[0].message.content.strip()
    else:result = eval(completion.choices[0].message.content.strip())
    print('    大模型调用成功！')
    
    return result
    
def post_processing_viz(result, img_path='temp/vl_now.jpg', check=False):
    
    '''
    视觉大模型输出结果后处理和可视化
    check：是否需要人工看屏幕确认可视化成功，按键继续或退出
    '''

    # 后处理
    img_bgr = cv2.imread(img_path)
    img_h = img_bgr.shape[0]
    img_w = img_bgr.shape[1]
    # 缩放因子
    FACTOR = 999
    # 起点物体名称
    START_NAME = result['start']
    # 起点，左上角像素坐标
    START_X_MIN = int(result['start_xyxy'][0][0] * img_w / FACTOR)
    START_Y_MIN = int(result['start_xyxy'][0][1] * img_h / FACTOR)
    # 起点，右下角像素坐标
    START_X_MAX = int(result['start_xyxy'][1][0] * img_w / FACTOR)
    START_Y_MAX = int(result['start_xyxy'][1][1] * img_h / FACTOR)
    # 起点，中心点像素坐标
    START_X_CENTER = int((START_X_MIN + START_X_MAX) / 2)
    START_Y_CENTER = int((START_Y_MIN + START_Y_MAX) / 2)



    
    # 可视化
    # 画起点物体框
    img_bgr = cv2.rectangle(img_bgr, (START_X_MIN, START_Y_MIN), (START_X_MAX, START_Y_MAX), [0, 0, 255], thickness=3)
    # 画起点中心点
    img_bgr = cv2.circle(img_bgr, [START_X_CENTER, START_Y_CENTER], 6, [0, 0, 255], thickness=-1)

    # 写中文物体名称
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb) # array 转 pil
    draw = ImageDraw.Draw(img_pil)
    # 写起点物体中文名称
    draw.text((START_X_MIN, START_Y_MIN-32), START_NAME, font=font, fill=(255, 0, 0, 1)) # 文字坐标，中文字符串，字体，rgba颜色
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # RGB转BGR
    print('    展示可视化效果图，按c键继续，按q键退出')
    # 保存可视化效果图
    cv2.imwrite('temp/vl_now_viz.jpg', img_bgr)

    # 在屏幕上展示可视化效果图
    cv2.imshow('zihao_vlm', img_bgr) 

    if check:
        print('    请确认可视化成功，按c键继续，按q键退出')
        while(True):
            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'): # 按c键继续
                cv2.destroyAllWindows()
                break
            if key == ord('q'): # 按q键退出
                exit()
    else:
        if cv2.waitKey(1) & 0xFF == None:
            pass

    return START_X_CENTER, START_Y_CENTER
    
    
def tcp_catch_lingyi(PROMPT = ''):
    print("tcp发送catch")
    fs = pipeline.wait_for_frames()
    fs = pipeline.wait_for_frames()
    fs = pipeline.wait_for_frames()
    aligned_frames = alignedFs.process(fs)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    #最终图像
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image_flipped_both = np.flip(color_image, (0, 1))
    cv2.imwrite('temp/vl_now.jpg', color_image_flipped_both)
    result = yi_vision_api(prompt=PROMPT, flag=1)
    START_X_CENTER, START_Y_CENTER = post_processing_viz(result, check=True)
    depth_pixel = [640-START_X_CENTER, 480-START_Y_CENTER]
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics# 获取相机参数
    dis = depth_frame.get_distance(int(640-START_X_CENTER), int(480-START_Y_CENTER))
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis) 
    #摄像头正前方是dz 越远离越大
    #摄像头左右是dx 越向右越大(反转了，所以向左为正)
    #摄像头上下是dy 越向下越大（反转了，所以向左为正）
    dx = -camera_coordinate[0]*100
    dy = -camera_coordinate[1]*100#符号为了调整成方向，调整之后恢复原来的描述
    dz = camera_coordinate[2]*100
    mystring = str(round(dx, 2))+" "+str(round(dy, 2))+" "+str(round(dz, 2))
    buffer_data = bytearray(17)
    buffer_data[0] = 0xFE
    buffer_data[1] = 0x04
    buffer_data[2] = 0x04#抓取动作
    buffer_data[3] = 0x0c
    uint_x = int(dx)
    uint_x += 9997
    print(uint_x)
    buffer_data[4] = uint_x%256
    buffer_data[5] = (uint_x>>8)%256
    buffer_data[6] = (uint_x>>16)%256
    buffer_data[7] = (uint_x>>24)%256
    uint_y = int(dy)
    uint_y += 10008
    buffer_data[8] = uint_y%256
    buffer_data[9] = (uint_y>>8)%256
    buffer_data[10] = (uint_y>>16)%256
    buffer_data[11] = (uint_y>>24)%256
    uint_z = int(dz)    
    uint_z += 10000
    buffer_data[12] = uint_z%256
    buffer_data[13] = (uint_z>>8)%256
    buffer_data[14] = (uint_z>>16)%256
    buffer_data[15] = (uint_z>>24)%256
    buffer_data[16] = 0x00
    i = 1
    while i < 16:
        buffer_data[16] = (buffer_data[16] + buffer_data[i])%256
        i+=1
    tcp.send_data(buffer_data)

    


def answer_lingyi(PROMPT=''):
    
    fs = pipeline.wait_for_frames()
    fs = pipeline.wait_for_frames()
    aligned_frames = alignedFs.process(fs)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    #最终图像
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image_flipped_both = np.flip(color_image, (0, 1))


    cv2.imwrite('temp/vl_now.jpg', color_image_flipped_both)
    cv2.imshow('zihao_vlm', color_image_flipped_both) 
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()   # 关闭所有opencv窗口
    print(66)
    result = yi_vision_api(prompt=PROMPT, flag=0)
    tts(result)
    play_wav('temp/tts.wav')
def tcp_release(PROMPT = ''):
    print("tcp发送relase")
    fs = pipeline.wait_for_frames()
    fs = pipeline.wait_for_frames()
    aligned_frames = alignedFs.process(fs)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    #最终图像
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    mat = cv2.Mat(color_image)
    #cv2.imwrite('temp/vl_now.jpg', mat)
    result = yi_vision_api(PROMPT,flag=1)
    START_X_CENTER, START_Y_CENTER = post_processing_viz(result, check=True)
    depth_pixel = [640-START_X_CENTER, 480-START_Y_CENTER]
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics# 获取相机参数
    dis = depth_frame.get_distance(int(640-START_X_CENTER), int(480-START_Y_CENTER))
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis) 
    #摄像头正前方是dz 越远离越大
    #摄像头左右是dx 越向右越大(反转了，所以向左为正)
    #摄像头上下是dy 越向下越大（反转了，所以向左为正）
    dx = -camera_coordinate[0]*100
    dy = -camera_coordinate[1]*100#符号为了调整成方向，调整之后恢复原来的描述
    dz = camera_coordinate[2]*100
    mystring = str(round(dx, 2))+" "+str(round(dy, 2))+" "+str(round(dz, 2))
    #cv2.putText(mat, mystring, (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
    buffer_data = bytearray(17)
    buffer_data[0] = 0xFE
    buffer_data[1] = 0x04
    buffer_data[2] = 0x05#放置动作
    buffer_data[3] = 0x0c
    uint_x = int(dx)
    uint_x += 10000
    print(uint_x)
    buffer_data[4] = uint_x%256
    buffer_data[5] = (uint_x>>8)%256
    buffer_data[6] = (uint_x>>16)%256
    buffer_data[7] = (uint_x>>24)%256
    uint_y = int(dy)
    uint_y += 10000
    buffer_data[8] = uint_y%256
    buffer_data[9] = (uint_y>>8)%256
    buffer_data[10] = (uint_y>>16)%256
    buffer_data[11] = (uint_y>>24)%256
    uint_z = int(dz)    
    uint_z += 10000
    buffer_data[12] = uint_z%256
    buffer_data[13] = (uint_z>>8)%256
    buffer_data[14] = (uint_z>>16)%256
    buffer_data[15] = (uint_z>>24)%256
    buffer_data[16] = 0x00
    i = 1
    while i < 16:
        buffer_data[16] = (buffer_data[16] + buffer_data[i])%256
        i+=1
    tcp.send_data(buffer_data)
    cv2.imshow("image", mat)
    cv2.waitKey(2000)
    
    is_release_finished = 0
    while(is_release_finished == 0):
        buffer_received = tcp.receive_data()
        print("接受到了数据")
        if buffer_received[0] == 0xFE and buffer_received[1] == 0x04 and buffer_received[2] == 0x00 and buffer_received[3] == 0x01:
            print("解包成功")
            i = 1
            numsum = 0
            while i < 5:
                numsum = (numsum + buffer_received[i])%256
                i+=1
            if numsum == buffer_received[5]:
                is_release_finished = buffer_received[4]
                print("放置完成")

def tcp_open():
    print("tcp发送open")
    buffer_data = bytearray(6)
    buffer_data[0] = 0xFE
    buffer_data[1] = 0x02
    buffer_data[2] = 0x02
    buffer_data[3] = 0x01
    buffer_data[4] = 0x00
    buffer_data[5] = 0x00


    buffer_data[4] = 0x01
    print("发送数据"+buffer_data)
    i = 1
    while i < 5:
        buffer_data[5] = (buffer_data[5] + buffer_data[i])%256
        i+=1
    tcp.send_data(buffer_data)
    is_open_finished = 0
    while(is_open_finished == 0):
        buffer_received = tcp.receive_data()
        print("接受到了数据")
        if buffer_received[0] == 0xFE and buffer_received[1] == 0x02 and buffer_received[2] == 0x00 and buffer_received[3] == 0x01:
            
            i = 1
            numsum = 0
            while i < 5:
                numsum = (numsum + buffer_received[i])%256
                i+=1
            print("解包成功")
            if numsum == buffer_received[5]:
                is_open_finished = buffer_received[4]
                print("成功打开")
def tcp_helloMove():
    print("tcp发送打招呼")
    buffer_data = bytearray(6)
    buffer_data[0] = 0xFE
    buffer_data[1] = 0x01
    buffer_data[2] = 0x01
    buffer_data[3] = 0x01
    buffer_data[4] = 0x00
    buffer_data[5] = 0x00


    buffer_data[4] = 0x01
    print("发送数据"+buffer_data)
    i = 1
    while i < 5:
        buffer_data[5] = (buffer_data[5] + buffer_data[i])%256
        i+=1
    tcp.send_data(buffer_data)
def tcp_dance():
    print("tcp发送catch")
    buffer_data = bytearray(6)
    buffer_data[0] = 0xFE
    buffer_data[1] = 0x00
    buffer_data[2] = 0x00
    buffer_data[3] = 0x01
    buffer_data[4] = 0x00
    buffer_data[5] = 0x00


    buffer_data[4] = 0x01
    print("发送数据"+buffer_data)
    i = 1
    while i < 5:
        buffer_data[5] = (buffer_data[5] + buffer_data[i])%256
        i+=1
    tcp.send_data(buffer_data)
    is_dance_finished = 0
    while(is_dance_finished == 0):
        buffer_received = tcp.receive_data()
        print("接受到了数据")
        if buffer_received[0] == 0xFE and buffer_received[1] == 0x00 and buffer_received[2] == 0x00 and buffer_received[3] == 0x01:
            
            i = 1
            numsum = 0
            while i < 5:
                numsum = (numsum + buffer_received[i])%256
                i+=1
            print("解包成功")
            if numsum == buffer_received[5]:
                is_dance_finished = buffer_received[4]
                print("跳完了，好累牙")
                
def app_agent(order):
    #arg_desc = "AppAgent - deployment phase"
    #parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
    #parser.add_argument("--app")
    #parser.add_argument("--root_dir", default="./")
    #args = vars(parser.parse_args())

    app = '电话'
    root_dir = './'

    print_with_color("Welcome to the deployment phase of AppAgent!\nBefore giving me the task, you should first tell me "
                 "the name of the app you want me to operate and what documentation base you want me to use. I will "
                 "try my best to complete the task without your intervention. First, please enter the main interface "
                 "of the app on your phone and provide the following information.", "blue")

    if not app:
        print_with_color("What is the name of the target app?", "blue")
        app = '电话'
        app = app.replace(" ", "")

    os.system(f"python scripts/task_executor.py --app {app} --root_dir {root_dir} --order {order}")
def pose():
    past_y = np.zeros(10)
    string_array = [""] * 10
    flag = 0
    seconds = 0
    x = 0
    y = 0
    while(1):#对每一帧的图像进行检测
        fs = pipeline.wait_for_frames()
        aligned_frames = alignedFs.process(fs)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        #最终图像
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image_flipped_both = np.flip(color_image, (0, 1))
        results = model_pose(color_image_flipped_both, tracker=True)
        annotated_frame = results[0].plot()
        count = 0
        x = 0
        y = 0
        if len(results[0].boxes.xywh) != 0:#假如其中有内容
            
            for xys in results[0].keypoints.xy:
                if xys[11][0]!=0 and xys[11][1]!=0:
                    depth_pixel = [xys[11][0], xys[11][1]]
                    x = xys[11][0]
                    y = xys[11][1]
                    x=640-x
                    y=480-y
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics# 获取相机参数
                    dis = depth_frame.get_distance(int(x), int(y))
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
                    if past_y[count] - camera_coordinate[1] > 0.1:
                        flag = 1
                    if flag == 1 and camera_coordinate[1] > 0.3:
                        seconds += 1
                    if seconds == 5:
                        string_array[count] = "fall"
                        app_agent('紧急求救')
                        seconds = 0
                        flag = 0
                    
                    past_y[count] = camera_coordinate[1]
                    cv2.putText(annotated_frame, string_array[count], (int(640-x),int(480-y)), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
                    count+=1
        else:
            for i in range(10):
                string_array[i] = ""

        cv2.imshow("image", annotated_frame)
        cv2.waitKey(50)
#摄像头正前方是dz 越远离越大
#摄像头左右是dx 越向右越大
#摄像头上下是dy 越向下越大
if __name__ == '__main__':
    #pose_thread = threading.Thread(target=pose, args=())
    #tcp_trace()
    UDP_IP = "10.78.76.37"
    UDP_PORT = 6666
    PROMPT="帮我抓取瓶子"
    result = yi_vision_api(prompt=PROMPT, flag=1)
    tts(result)
    pose()
    #video_thread = threading.Thread(target=video_sender, args=(UDP_IP, UDP_PORT))
    #video_thread.start()
    tcp_trace("person")

    while(1):
        start_record_ok = input('是否开启录音，按r开始录制，按k打字输入，按c输入默认指令')
        if start_record_ok == 'r':
            record()   # 录音
            order = speech_recognition() # 语音识别
        elif start_record_ok == 'k':
            order = input('请输入指令')
        elif start_record_ok == 'c':
            order = '请帮我抓取桌子上的杯子'
        # 智能体Agent编排动作
        agent_plan_output = eval(agent_plan(order))
    
        print('智能体编排动作如下\n', agent_plan_output)
        plan_ok = input('是否继续？按c继续，按q退出')
        plan_ok = 'c'
        if plan_ok == 'c':
            response = agent_plan_output['response'] # 获取机器人想对我说的话
            tts(response)                     # 语音合成，导出wav音频文件
            play_wav('temp/tts.wav')          # 播放语音合成音频文件
            for each in agent_plan_output['function']: # 运行智能体规划编排的每个函数
                print('开始执行动作', each)
                eval(each)
        elif plan_ok =='q':
            exit()
