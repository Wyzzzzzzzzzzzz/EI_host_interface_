import cv2
import pyrealsense2 as rs
# 打开摄像头，0通常是默认的内置摄像头
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
align_to = rs.stream.color# 设定需要对齐的方式（这里是深度对齐彩色，彩色图不变，深度图变换）
alignedFs = rs.align(align_to)
profile = pipeline.start(cfg)


# 逐帧读取摄像头数据
while True:
    fs = pipeline.wait_for_frames()
    aligned_frames = alignedFs.process(fs)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    # ret是一个布尔值，表示是否成功读取帧


    # 显示帧
    cv2.imshow('Frame', color_frame)

    # 按下 's' 键时，保存图片
    if cv2.waitKey(1) == ord('s'):
        # 保存图片到当前目录下，文件名为'image.png'
        cv2.imwrite('image.png', color_frame)

    # 按下 'q' 键时，退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头资源

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
