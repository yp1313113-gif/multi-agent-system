# 常用命令（Windows 请用 Git Bash / WSL）

install:
	pip install -r requirements.txt

# 构建知识库（需先放好模型权重 models/bge-reranker-v2-m3，及 bge-small-zh 会自动下载/本地加载）
ingest:
	python ingest.py

# 跑测试（默认跳过需要模型权重的用例）
test:
	MODEL_TESTS=0 pytest -q tests/

# 本地交互式运行
run:
	python run.py

# 启动 FastAPI 服务（访问 http://localhost:8000/chat?message=你好）
serve:
	uvicorn api:app --host 0.0.0.0 --port 8000

# Docker 一键构建+启动（含 Chroma + Redis）
docker:
	docker-compose up --build

# 多源隔离逻辑验证（无需模型权重，用假 embedding）
verify-multisource:
	python verify_multisource.py

.PHONY: install ingest test run serve docker verify-multisource
