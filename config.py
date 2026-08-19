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
            "考勤与假期": {
                "description": "考勤与假期制度（工作时间、迟到早退、加班、年假/病假/事假/婚假/产假/陪产/丧假）",
                "chapters": [2, 3],
            },
            "薪酬与福利": {
                "description": "薪酬福利与出差报销（薪酬结构、工资发放、五险一金、补贴、出差申请/差旅标准/报销流程）",
                "chapters": [4, 5],
            },
            "通用制度": {
                "description": "公司通用制度（总则、培训发展、行为规范、附则）",
                "chapters": [1, 6, 7, 8],
            },
        }
        self.DEFAULT_DATA_SOURCE = os.getenv("DEFAULT_DATA_SOURCE", "考勤与假期")
        # 共享集合名（ingest 与检索两端必须一致）
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_handbook")
        
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