
#!/usr/bin/env bash

curl -s -X POST 'http://127.0.0.1:5001/submit' \
	-H 'Content-Type: application/json' \
	-d @- <<'JSON' | jq
{
	"lmdeploy_api_key": "sk-73f3e76569654254aad1cd02be77638d",
	"lmdeploy_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
	"lmdeploy_model_name": "qwen3-vl-plus",
	"judge_api_key": "sk-73f3e76569654254aad1cd02be77638d",
	"judge_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
	"judge_model_name": "qwen2.5-7b-instruct",
	"LMUData": "/home/ubuntu/VLMEvalKit/LMUData",
	"data": "MathVista_MINI_10",
	"extra_args": {
		"dry_run": false
	}
}
JSON


curl -s -X POST 'http://118.25.8.159:5001/submit' \
	-H 'Content-Type: application/json' \
	-d @- <<'JSON' | jq
{
	"lmdeploy_api_key": "sk-73f3e76569654254aad1cd02be77638d",
	"lmdeploy_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
	"lmdeploy_model_name": "qwen3-vl-plus",
	"judge_api_key": "sk-73f3e76569654254aad1cd02be77638d",
	"judge_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
	"judge_model_name": "qwen2.5-7b-instruct",
	"LMUData": "/home/ubuntu/VLMEvalKit/LMUData",
	"data": "MathVista_MINI_10",
	"extra_args": {
		"dry_run": false
	}
}
JSON


# echo "--- Checking status for existing job '3f0e4a0c' ---"
# curl -s http://127.0.0.1:5001/status/3f0e4a0c | python -m json.tool


# # 启动服务
pids=$(ss -ltnp 2>/dev/null | awk '/:5001 /{match($0,/pid=([0-9]+)/,a); if(a[1]) print a[1]}'); if [ -n "$pids" ]; then echo "Stopping existing service (PIDs: $pids)..."; echo "$pids" | xargs -r -n1 kill -9 && sleep 0.6; fi;
nohup python -u -c "import importlib, flask_service.app as a; importlib.reload(a); a.app.run(host='0.0.0.0', port=5001)" > flask_service.log 2>&1 &
echo "Flask service starting in background. PID: $!"
sleep 1
ss -ltnp | grep 5001 || echo "Service may have failed to start. Check flask_service.log"



# # 停止服务
pids=$(ss -ltnp 2>/dev/null | awk '/:5001 /{match($0,/pid=([0-9]+)/,a); if(a[1]) print a[1]}'); if [ -n "$pids" ]; then echo "Stopping Flask service (PID: $pids)..."; echo "$pids" | xargs -r -n1 kill -9 && echo "Service stopped."; else echo "Service not running."; fi



curl -s "http://127.0.0.1:5001/status/3bf12d59" | jq .

curl -s "http://127.0.0.1:5001/results/7f61213c" | jq .



JOB=7f61213c
mkdir -p dl/$JOB
curl -s "http://127.0.0.1:5001/results/$JOB" \
  | jq -r '.files[]' \
  | while read file; do
      curl -s -G "http://127.0.0.1:5001/download/$JOB" --data-urlencode "path=$file" -o "dl/$JOB/$(basename "$file")"
    done


sudo docker run -d \
  --name vlmeval-service \
  -p 5001:5001 \
  -v "$(pwd)/outputs":/app/outputs \
  vlmeval-service

 docker exec -it vlmeval-service /bin/bash  


  # 构建并启动
docker-compose up -d --build

# 查看容器日志（实时）
docker-compose logs -f

# 停止并移除容器（和网络）
docker-compose down

