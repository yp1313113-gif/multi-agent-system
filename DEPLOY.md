# 部署指南

本服务是一个 FastAPI 应用，提供 `/chat` 流式接口，依赖：**向量库（Chroma）**、**Redis（缓存，可降级）**、**DeepSeek API（生成）**。
reranker 模型（`bge-reranker-v2-m3`）缺失时系统会**自动降级为无重排模式**，仍可运行，只是召回精度略低。

---

## 一、本地 / 自有服务器（最简单）

```bash
git clone https://github.com/yp1313113-gif/multi-agent-system.git
cd multi-agent-system
pip install -r requirements.txt

# 1) 准备模型权重（二选一）
#    a. 让 bge-small-zh-v1.5 走 HF 镜像自动下载（去掉 local_files_only 或联网）；
#    b. 把 bge-reranker-v2-m3 放到 ./models/bge-reranker-v2-m3/（含 config.json）。
# 2) 配置 .env（至少 DEEPSEEK_API_KEY）
# 自建 .env，填入 DEEPSEEK_API_KEY
# 3) 构建知识库
python ingest.py
# 4) 启动
uvicorn api:app --host 0.0.0.0 --port 8000
```

访问：`http://<你的IP>:8000/chat?message=研发费用加计扣除比例是多少？`

---

## 二、Docker（推荐，含 Chroma + Redis）

```bash
docker-compose up --build
```

> 注意：Docker 镜像内如需使用 reranker，请在构建前把 `models/bge-reranker-v2-m3` 放好（或调整 Dockerfile 让其下载）。
> 首次运行后需在容器内执行 `python ingest.py` 构建向量库（或挂载已构建的 chroma_db 卷）。

---

## 三、公开可演示（给面试官/HR 看的链接）

要让外网能访问，需要一个云平台账号（**这一步需要你自己的账号，我无法直接代你发布到公网**）：

| 平台 | 适用 | 说明 |
|------|------|------|
| **Railway / Render** | FastAPI 长驻服务 | 连接 GitHub 仓库自动部署，免费额度够 demo；`uvicorn api:app` 作为启动命令 |
| **Vercel** | 仅适合 Serverless（本服务有状态/长连接，不首选） | — |
| **阿里云 / 腾讯云 函数 + API 网关** | 国内低延迟 | 需打包依赖，略复杂 |

**最省事路径**：把仓库推到 GitHub，去 Railway 选 "Deploy from GitHub"，启动命令填 `uvicorn api:app --host 0.0.0.0 --port $PORT`，
再把生成的 `*.up.railway.app` 链接写进简历。这样面试官一点就能对话，比截图有说服力。

> 沙箱环境无法把服务暴露到公网（你的浏览器也连不上沙箱 localhost），所以公开部署必须由你用自己的账号完成；我可帮你把配置/启动命令/README 准备妥当。

---

## 四、健康检查与验证

- 本地起服务后：`curl "http://localhost:8000/chat?message=研发费用加计扣除比例是多少？"`
- 检索指标复现：`python eval/retrieval_eval.py`
