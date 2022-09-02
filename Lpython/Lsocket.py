
import socket
# 函数 gethostname() 返回当前正在执行 Python 的系统主机名
res = socket.gethostbyname(socket.gethostname())

print(res)