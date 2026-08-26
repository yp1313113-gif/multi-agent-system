# config.py
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self._env_path = Path(".env")
        self._last_mtime = self._get_file_mtime()
        self._load()
    
    def _get_file_mtime(self):
        if self._env_path.exists():
            return self._env_path.stat().st_mtime
        return 0
    
    def _load(self):
        # ---- LLM ----
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
        
        # ---- Agent ----
        self.RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "10"))
        
        # ---- 工具 ----
        self.TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "10"))
        self.TOOL_MAX_RETRIES = int(os.getenv("TOOL_MAX_RETRIES", "3"))
        self.TOOL_RETRY_DELAY = int(os.getenv("TOOL_RETRY_DELAY", "1"))
        
        # ---- 并发控制 ----
        self.MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "10"))
        self.CONCURRENCY_QUEUE_TIMEOUT = float(os.getenv("CONCURRENCY_QUEUE_TIMEOUT", "5"))
        
        # ---- Redis 缓存 ----
        self.CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
        self.REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))
        
        # ---- Langfuse 可观测性 ----
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://jp.cloud.langfuse.com")
        
        # ---- HITL ----
        self.HITL_ENABLED = os.getenv("HITL_ENABLED", "true").lower() == "true"
        self.HITL_HIGH_RISK_TOOLS = os.getenv("HITL_HIGH_RISK_TOOLS", "rag_search").split(",")
        
        # ---- 数据源（逻辑隔离）----
        # 所有数据源共享同一个 Chroma 集合，每块切片用 metadata["source"] 标记；
        # "切换知识库" = 查询时按 source 做元数据过滤（where={"source": x}）。
        # chapters 字段用于 ingest 阶段按章节把手册切分到对应数据源。
        self.DATA_SOURCES = {
            "研发费用政策库": {
                "description": "研发费用加计扣除政策（100%加计、六大费用口径、高企认定条件、不适用情形）",
            },
            "研发费用归集FAQ": {
                "description": "研发费用归集常见问答（研发活动判断、辅助账、留存备查资料、申报时间）",
            },
            "研发费用风险指标库": {
                "description": "金四对标风险指标（人员/直接投入/折旧/其他/综合）",
            },
        }
        self.DEFAULT_DATA_SOURCE = os.getenv("DEFAULT_DATA_SOURCE", "研发费用政策库")
        # 共享集合名（ingest 与检索两端必须一致）
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rd_expense")
        
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    def reload(self):
        old_model = self.DEEPSEEK_MODEL
        old_temp = self.TEMPERATURE
        old_limit = self.RECURSION_LIMIT
        self._load()
        print(f"\n🔄 配置已更新")
        print(f"   DEEPSEEK_MODEL: {old_model} → {self.DEEPSEEK_MODEL}")
        print(f"   TEMPERATURE: {old_temp} → {self.TEMPERATURE}")
        print(f"   RECURSION_LIMIT: {old_limit} → {self.RECURSION_LIMIT}")
    
    def check_and_reload(self):
        current_mtime = self._get_file_mtime()
        if current_mtime != 0 and current_mtime != self._last_mtime:
            self._last_mtime = current_mtime
            self.reload()
            return True
        return False

config = Config()