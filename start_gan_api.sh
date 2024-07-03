source ./venv/bin/activate

port=9091

# 检查port是否已读取
if [ -z "$port" ]; then
  echo "Port not found in configuration file."
  exit 1
fi

echo "Looking for processes using TCP port $port..."

# 使用lsof命令获取特定TCP端口的进程PID
pid=$(lsof -t -i tcp:$port)

# 检查PID是否为空，如果不为空，则输出PID
if [ ! -z "$pid" ]; then
  echo "Found process on port $port with PID: $pid"
  # 如果需要杀死该进程，取消下一行的注释
  kill -9 $pid
else
  echo "No process found on port $port."
fi

sleep 2

nohup python -u gan_api.py --dataroot '' > nohub.log 2>&1 &

echo "start success!!"
