# tools/weather_tool.py
import requests
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_fixed
from exceptions import ToolTimeoutError, ToolRateLimitError
from config import config

@retry(stop=stop_after_attempt(config.TOOL_MAX_RETRIES), wait=wait_fixed(config.TOOL_RETRY_DELAY))
def _weather_query(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=%C+%t&lang=zh"
        response = requests.get(url, timeout=config.TOOL_TIMEOUT)
        if response.status_code != 200:
            raise ToolRateLimitError(f"天气API返回状态码: {response.status_code}")
        return f"{city}天气: {response.text.strip()}"
    except requests.exceptions.Timeout:
        raise ToolTimeoutError(f"天气API超时（{config.TOOL_TIMEOUT}秒）")
    except requests.exceptions.ConnectionError:
        raise ToolRateLimitError("网络连接失败")

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    try:
        return _weather_query(city)
    except ToolTimeoutError as e:
        return f"⏰ {str(e)}，请稍后重试"
    except ToolRateLimitError as e:
        return f"⚠️ {str(e)}"
    except Exception as e:
        return f"❌ 天气查询失败: {e}"