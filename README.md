# 轨道交通领域检索与故障辅助诊断 Agent

一个可复现、可评测且遵守语料授权边界的轨道交通 RAG 原型。项目覆盖公开来源登记、
PDF/HTML/ZIP 解析、BM25 与向量双路召回、RRF 融合、Cross-Encoder 重排、检索评测，
以及带原文证据和来源定位的回答生成。当前语料为 18 份文档、949 个结构化切片，来自
16 个启用来源。

## 工作内容

- 以 `source_manifest.yaml` 登记来源、版本、许可证和仓库策略；
- 重建本地原始语料。
- 解析和清洗 PDF、扫描 PDF、HTML 与指定 ZIP 内容，输出统一的 `documents.jsonl` 与
  `chunks.jsonl`，保留页码、章节、来源 URL 和内容哈希。
- 完成 BM25、`multilingual-e5-small` 向量检索、RRF 融合和 mMARCO Cross-Encoder
  重排；针对长中文切片使用 query-aware window，避免 512-token 截断答案片段。
- 建立 44 条人工整理的问题规格（中文 30、英文 14），将参考答案逐条绑定到来源、
  文档、页码或网页章节及证据切片，并计算 Recall@K、Hit@K、MRR、nDCG 与 P95 延迟。
- 提供命令行检索与问答；有 API key 时通过 OpenAI Responses API 生成严格基于证据的
  回答，无 key 时使用确定性的抽取式回答。

## 最终评测

本地 Apple Silicon CPU；949 个切片；44 个问题。模型首次加载和语料编码时间不计入
单次查询延迟，完整结果见 `evaluation/results.json`。

| 方案 | Recall@5 | Recall@20 | MRR@10 | nDCG@10 | P95 延迟 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.966 | 0.977 | 0.835 | 0.868 | 1.46 ms |
| E5 向量检索 | 0.920 | 1.000 | 0.755 | 0.798 | 18.09 ms |
| BM25 + E5 + RRF | 0.966 | 1.000 | 0.875 | **0.901** | **18.44 ms** |
| RRF + Cross-Encoder | 0.960 | 1.000 | **0.876** | 0.891 | 1473.14 ms |

RRF 在当前语料上取得更好的综合排序质量和 CPU 延迟，因此 CLI 默认采用 `hybrid`。
Cross-Encoder 略微提高 MRR，但降低 nDCG 且显著增加延迟，作为可选消融阶段保留，
而不是预设重排一定有效。该结果来自小规模内部测试集，不代表通用行业基准；正式公开
前仍应由轨道交通领域人员独立复核问题和答案。

## 快速开始

运行环境建议为 macOS/Linux 与 Python 3.9+。脚本会在项目内创建 `.venv` 并安装依赖。

```bash
# 1. 检查本地状态
./scripts/rail_agent.sh doctor

# 2. 重建原始语料和结构化切片
./scripts/rebuild_raw_corpus.sh
./scripts/build_processed_corpus.sh

# 3. 将问题规格重新绑定到当前切片
./scripts/build_evaluation_set.py

# 4. 首次下载模型并建立向量缓存；完成后自动使用离线缓存
./scripts/rail_agent.sh index --with-reranker

# 5. 运行四阶段评测
./scripts/rail_agent.sh evaluate --modes bm25,dense,hybrid,rerank
```

## 命令行演示

检索会返回标题、来源 ID、页码或章节、切片 ID、得分与原文预览：

```bash
./scripts/rail_agent.sh search \
  '关键设施设备的检修记录需要保存多久？' \
  --mode hybrid --top-k 5
```

无 API key 也可以得到带 `[S1]` 引用、来源链接、定位信息和原文证据的抽取式答案：

```bash
./scripts/rail_agent.sh ask \
  '关键设施设备的检修记录需要保存多久？' \
  --mode hybrid --llm never --top-k 5
```

如需大模型生成，在当前 shell 中配置 key；不要把 key 写入仓库：

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='gpt-5-mini'
./scripts/rail_agent.sh ask \
  '关键设施设备的检修记录需要保存多久？' \
  --mode hybrid --llm required
```

若只拿到发布 ZIP 而未下载完整语料，可以对包内开放许可样例运行英文演示：

```bash
./scripts/rail_agent.sh \
  --chunks demo_data/chunks.sample.jsonl \
  search 'How is the RCM-DX data format specified?' --mode bm25
```

## 数据重建选项

```bash
# 仅验证 manifest 或查看计划，不联网
./scripts/rebuild_raw_corpus.sh --validate-only
./scripts/rebuild_raw_corpus.sh --dry-run

# 仅下载开放许可来源，或只重建一个来源
./scripts/rebuild_raw_corpus.sh --open-only
./scripts/rebuild_raw_corpus.sh --only cn_mot_maintenance_2024

# 默认关闭约 208 MB 的 MetroPT-3；需要时显式启用
./scripts/rebuild_raw_corpus.sh --include-disabled

# 调整切片大小
./scripts/build_processed_corpus.sh --chunk-size 900 --chunk-overlap 120
```

## 目录结构

```text
configs/                 检索、模型、生成与评测参数
data/raw/                本地原始材料（Git 忽略）
data/processed/          documents.jsonl / chunks.jsonl（Git 忽略）
data/cache/              模型与向量缓存（Git 忽略）
docs/                    架构说明
evaluation/              问题规格、可发布标签与评测结果
scripts/                 下载、解析、评测、CLI 与打包入口
src/rail_agent/          检索、融合、重排、指标与回答生成代码
tests/                   无网络单元测试
source_manifest.yaml     可追溯来源与许可策略
```

## 合规边界

- `manifest_only`：只发布 manifest、来源链接、短参考答案和代码，不发布原始或完整衍生
  正文。
- `raw_allowed_with_attribution`：许可证允许复用，但必须保留署名、许可证与来源链接。
- `raw_allowed_noncommercial_sharealike`：仅限非商业场景，衍生内容需遵守相同许可。
- `raw_allowed_with_third_party_exclusions`：排除第三方图片、附件、标识与商标。

`.gitignore` 同时排除 `data/raw/`、`data/processed/`、`data/cache/`、`.venv/` 与
`artifacts/`。ZIP 打包器再次使用 allowlist 检查路径，形成第二道防线。

## 构建作品包

```bash
./scripts/build_release_zip.sh
```

输出为 `artifacts/rail-transit-retrieval-agent.zip`。50–100 MB 通常是上传大小上限，
不是必须达到的体积；本项目不应通过塞入模型或受限 PDF 人为增大压缩包。

架构细节见 `docs/architecture.md`，问题集与指标口径见 `evaluation/README.md`。
