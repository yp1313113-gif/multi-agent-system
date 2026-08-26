# ingest.py
"""知识库构建脚本（多文件 + 清洗 + 结构切分 + 元数据 source=文件名）。

处理流水线：
  原始文本 → ① 数据清洗(clean_text) → ② 递归字符切块（以 ## / ### 标题为结构边界，带重叠）
  → ③ 打元数据（source=文件名、chunk_index） → ④ bge-small-zh 向量化写入 Chroma。

多源逻辑隔离：每个 txt 文件 = 一个逻辑数据源（metadata["source"]=文件名），
检索阶段通过 where={"source": x} 切换知识库，不破坏召回质量。
"""
import os
import sys
import argparse
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import config
from tools.data_clean import clean_text


def build_documents(raw: str, source: str):
    """清洗后返回单篇文档（source 为逻辑数据源名）。"""
    cleaned = clean_text(raw)
    return [Document(page_content=cleaned, metadata={"source": source})]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", default="./data", help="数据文件夹路径")
    parser.add_argument("--output", "-o", default="./chroma_db", help="向量库保存路径")
    parser.add_argument("--chunk-size", type=int, default=500, help="切块大小")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="切块重叠")
    args = parser.parse_args()

    files = sorted(Path(args.path).glob("*.txt"))
    if not files:
        print(f"❌ data 目录下没有 txt 文件: {args.path}")
        sys.exit(1)

    all_docs = []
    for fp in files:
        source = fp.stem  # 文件名（去 .txt）作为逻辑数据源
        raw = TextLoader(str(fp), encoding='utf-8').load()[0].page_content
        docs = build_documents(raw, source)
        all_docs.extend(docs)
        print(f"✅ 已加载 [{source}]（{len(raw)} 字符）")

    print(f"✂️ 正在结构切块（chunk_size={args.chunk_size}，overlap={args.chunk_overlap}）...")
    # 以 ## / ### 标题作为切分边界（相当于按结构切分），再递归字符切块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        keep_separator=True,
    )
    chunks = text_splitter.split_documents(all_docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    print(f"✅ 切成 {len(chunks)} 个文本块（含 source/chunk_index 元数据）")

    stat = {}
    for c in chunks:
        s = c.metadata.get("source", "未知")
        stat[s] = stat.get(s, 0) + 1
    for s, n in stat.items():
        print(f"   · [{s}]: {n} 块")

    print("🔢 正在向量化...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"💾 正在写入向量库: {args.output}（集合: {config.COLLECTION_NAME}）")
    Chroma.from_documents(chunks, embeddings, persist_directory=args.output, collection_name=config.COLLECTION_NAME)
    print(f"✅ 向量库构建完成！保存在: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
