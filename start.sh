source ./venv/bin/activate

# 指定配置文件路径
config_file="config.ini"

# 从配置文件中读取信息
port=$(awk -F' = ' '/^port/ {print $2}' <(grep 'port' $config_file))

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

nohup python -u webui.py > nohub.log 2>&1 &

echo "start success!!"
