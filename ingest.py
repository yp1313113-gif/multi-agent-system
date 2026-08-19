# ingest.py
"""
知识库构建脚本（多源逻辑隔离 + 数据清洗 + 分级切块）。

处理流水线：
  原始文本 → ① 数据清洗(clean_text) → ② 按「章」切分并归属逻辑数据源
  → ③ 每章内按「小节(###)」细分 → ④ 递归字符切块(带重叠)
  → ⑤ bge-small-zh 向量化写入同一 Chroma 集合。

多源逻辑隔离：所有源写入【同一个 Chroma 集合】，每块切片用
metadata["source"] 标记；检索阶段通过 `where={"source": x}` 实现“切换知识库”，
不破坏召回质量。
"""
import os
import re
import argparse
import sys
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import config
from tools.data_clean import clean_text
from tools.doc_parser import parse_chapters, split_sections, chapter_to_source


def build_documents(raw: str):
    """清洗 → 切章 → 切小节 → 打元数据，返回 LangChain Document 列表。"""
    cleaned = clean_text(raw)
    chapters = parse_chapters(cleaned)
    print(f"✅ 清洗后解析出 {len(chapters)} 个章节")

    docs = []
    for num, body in chapters:
        src = chapter_to_source(num)
        for sec_title, sec_body in split_sections(body):
            meta = {"source": src, "chapter": num}
            if sec_title:
                meta["section"] = sec_title
            docs.append(Document(page_content=sec_body, metadata=meta))
            sec_disp = f" / {sec_title}" if sec_title else ""
            print(f"   · 第{num}章{sec_disp} -> 数据源「{src}」")
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", default="./data", help="数据文件夹路径（含公司行政管理手册.txt）")
    parser.add_argument("--output", "-o", default="./chroma_db", help="向量库保存路径")
    parser.add_argument("--chunk-size", type=int, default=800, help="切块大小")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="切块重叠")
    args = parser.parse_args()

    manual_path = os.path.join(args.path, "公司行政管理手册.txt")
    if not os.path.exists(manual_path):
        print(f"❌ 找不到手册: {manual_path}")
        sys.exit(1)

    print(f"📄 正在加载手册: {manual_path}")
    raw = TextLoader(manual_path, encoding='utf-8').load()[0].page_content

    docs = build_documents(raw)

    print(f"✂️ 正在切块（chunk_size={args.chunk_size}，overlap={args.chunk_overlap}）...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        keep_separator=True,
    )
    chunks = text_splitter.split_documents(docs)
    # split_documents 会继承父文档的 metadata（source/chapter/section 自动带到每个切片）
    # 再补一个全局序号，方便问题溯源
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    print(f"✅ 切成 {len(chunks)} 个文本块（含 source/chapter/section 元数据）")

    sources_stat = {}
    for c in chunks:
        s = c.metadata.get("source", "未知")
        sources_stat[s] = sources_stat.get(s, 0) + 1
    for s, n in sources_stat.items():
        print(f"   · 「{s}」: {n} 块")

    print("🔢 正在向量化...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"💾 正在写入向量库: {args.output}（集合: {config.COLLECTION_NAME}）")
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=args.output,
        collection_name=config.COLLECTION_NAME,
    )
    print(f"✅ 向量库构建完成！保存在: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
