import threading
import time

# 定义第一个线程要运行的函数
def print_numbers():
    for i in range(1, 6):
        print(f"Thread 1: {i}")
        time.sleep(1)

# 定义第二个线程要运行的函数
def print_letters():
    while 1:
        print(1111)
        time.sleep(1)
if __name__ == '__main__':
    # 创建线程
    thread1 = threading.Thread(target=print_numbers)
    thread2 = threading.Thread(target=print_letters)

    # 启动线程
    thread1.start()
    thread2.start()
    string = "print_letters"
    # 等待所有线程完成
    eval()

    print("Both threads have finished execution.")
