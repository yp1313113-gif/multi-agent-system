# eval/ragas_eval.py
"""
RAG 质量评测（Ragas）。

评估维度：
  - Faithfulness（忠实度）：答案是否完全由检索到的上下文支撑，无幻觉；
  - AnswerRelevancy（答案相关性）：答案是否切题。

用法：
  1. 先构建知识库：python ingest.py
  2. 安装评测依赖：pip install ragas datasets
  3. 运行：python eval/ragas_eval.py
     （默认使用 scripts/testset.py 里的 12 条样本；也可改成读自己的标注集）

说明：本脚本调用 DeepSeek 作为生成模型、bge-small-zh 作为 embedding，
      与线上链路一致；评测结果用于量化「检索→生成」质量，是 Eval 体系的一部分。
"""
import os
import sys

# 让脚本能 import 项目根模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config
from tools.rag_tool import get_retriever


# 评测样本：问题 + 期望回答所依赖的真实知识点（reference，用于 faithfulness/answer_relevancy 计算）
TESTSET = [
    {"question": "年假怎么算？工作刚满1年能休几天？",
     "reference": "职工累计工作已满1年不满10年的，年休假5天。"},
    {"question": "迟到和早退怎么处理？",
     "reference": "迟到或早退按次数扣减当月全勤奖，并计入考勤记录。"},
    {"question": "加班有加班费吗？",
     "reference": "工作日加班支付不低于工资150%的报酬，休息日200%，法定节假日300%。"},
    {"question": "病假需要提供什么材料？",
     "reference": "病假须提供正规医疗机构出具的病假证明。"},
    {"question": "婚假是多少天？",
     "reference": "依法登记结婚享受婚假，具体天数按当地规定执行。"},
    {"question": "工资什么时候发？",
     "reference": "工资于每月固定日期发放，遇节假日提前。"},
    {"question": "五险一金包含哪些？",
     "reference": "五险一金包括养老、医疗、失业、工伤、生育保险及住房公积金。"},
    {"question": "出差怎么申请？",
     "reference": "出差须提前在系统提交出差申请，经审批后执行。"},
    {"question": "差旅住宿标准是多少？",
     "reference": "差旅住宿按职级对应标准执行，超标部分自理。"},
    {"question": "报销流程是什么？",
     "reference": "报销须填写报销单并附合规票据，经审核后财务支付。"},
    {"question": "有哪些培训机会？",
     "reference": "公司提供入职培训、岗位技能与管理能力培训。"},
    {"question": "行为规范里对保密有什么要求？",
     "reference": "员工须保守公司及客户商业秘密，不得泄露。"},
]


def build_eval_dataset():
    """用真实检索链路生成答案，构造 Ragas 所需数据集。"""
    retriever = get_retriever()
    rows = {"question": [], "answer": [], "contexts": [], "reference": []}
    for item in TESTSET:
        q = item["question"]
        docs = retriever.hybrid_search(q, source=None, top_k=3)
        contexts = [d.page_content for d in docs]
        # 用与线上一致的 prompt 生成答案
        from tools.rag_tool import rag_chain
        answer = rag_chain.invoke({"docs": docs, "question": q})
        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["reference"].append(item["reference"])
        print(f"  · 已生成: {q[:20]}... (上下文 {len(contexts)} 块)")
    return Dataset.from_dict(rows)


def main():
    print("📊 开始 Ragas 评测（Faithfulness / AnswerRelevancy）...")
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    ))

    dataset = build_eval_dataset()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )
    print("\n✅ 评测完成：")
    print(result)
    # 同时落一份 CSV，便于对比迭代
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_result.csv")
    result.to_pandas().to_csv(out, index=False)
    print(f"📄 详细结果已保存: {out}")


if __name__ == "__main__":
    main()
