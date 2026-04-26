import os
import re
import math
from collections import Counter
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer, CrossEncoder


class BGEM3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self._model = model

    def embed_documents(self, texts):
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = self._model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

    def __call__(self, input):
        return self.embed_documents(input)


# ============================================================
# BM25 实现（纯 Python，无额外依赖）
# ============================================================

class BM25:
    """
    轻量级 BM25 实现，专为代码 + 文档混合语料优化。
    Tokenizer 会同时保留：
      - 驼峰/Pascal 词拆分（RegInit -> Reg Init）
      - 下划线分词（io_out -> io out）
      - Chisel/Scala 关键字原词（保留 := 等操作符）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.doc_freqs: list[dict] = []
        self.idf: dict = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    # ----------------------------------------------------------
    # Tokenizer：兼顾自然语言 + 代码标识符
    # ----------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = []

        # 1. 保留 Chisel/Scala 操作符作为整体 token
        chisel_ops = [":=", "<>", "->", "=>"]
        for op in chisel_ops:
            text = text.replace(op, f" {op} ")

        # 2. 驼峰拆分（RegInit -> Reg Init, UInt -> U Int 不合适，所以只拆大写开头的单词边界）
        def split_camel(word):
            return re.sub(r"([a-z])([A-Z])", r"\1 \2", word).split()

        # 3. 按空白和非字母数字分割（保留 _ 连接词先整体处理）
        raw_tokens = re.findall(r"[A-Za-z0-9_:=<>]+|[^\s]", text)

        for tok in raw_tokens:
            if tok in chisel_ops:
                tokens.append(tok)
                continue
            # 下划线分词
            parts = tok.split("_")
            for part in parts:
                if not part:
                    continue
                # 驼峰拆分
                sub = split_camel(part)
                tokens.extend(sub)

        # 4. 小写化，过滤纯数字和长度 < 2 的 token
        result = []
        for t in tokens:
            t_lower = t.lower()
            if len(t_lower) >= 2:
                result.append(t_lower)
        return result

    # ----------------------------------------------------------
    # 构建索引
    # ----------------------------------------------------------
    def fit(self, corpus: list[str]):
        self.corpus = corpus
        self.N = len(corpus)
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]

        # 文档频率
        df = Counter()
        for tokens in self.tokenized_corpus:
            for tok in set(tokens):
                df[tok] += 1

        # IDF（Robertson IDF，防止除零）
        self.idf = {}
        for tok, freq in df.items():
            self.idf[tok] = math.log(
                (self.N - freq + 0.5) / (freq + 0.5) + 1
            )

        # 词频字典列表
        self.doc_freqs = [Counter(tokens) for tokens in self.tokenized_corpus]

        # 平均文档长度
        self.avgdl = sum(len(t) for t in self.tokenized_corpus) / self.N if self.N else 1.0

    def get_scores(self, query: str) -> list[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.N

        for tok in query_tokens:
            if tok not in self.idf:
                continue
            idf_val = self.idf[tok]
            for i, freq_dict in enumerate(self.doc_freqs):
                tf = freq_dict.get(tok, 0)
                if tf == 0:
                    continue
                dl = len(self.tokenized_corpus[i])
                norm = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                scores[i] += idf_val * norm

        return scores


# ============================================================
# 混合检索 RAG 服务
# ============================================================

class ChiselHybridRAGService:
    """
    混合检索 RAG 服务：
      1. 向量检索（语义）  → 召回候选
      2. BM25 检索（关键字）→ 召回候选
      3. RRF 融合           → 合并排名
      4. Reranker 精排      → 最终排序
    """

    def __init__(
        self,
        db_path: str = "/root/chisel-RAG/vector_db",
        embed_model_path: str = "/root/.cache/modelscope/hub/models/BAAI/bge-m3",
        rerank_model_path: str = "/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ):
        print(f"正在加载混合检索 RAG 服务 (Device: {device})...")
        self.device = device
        self.rrf_k = rrf_k

        # 1. Embedding 模型
        print("Loading Embedding Model...")
        self.embed_model = SentenceTransformer(embed_model_path, device=device)
        self.embedding_function = BGEM3EmbeddingFunction(self.embed_model)

        # 2. Reranker
        print("Loading Reranker Model...")
        try:
            self.reranker = CrossEncoder(rerank_model_path, device=device, trust_remote_code=True)
            self.use_rerank = True
            print("Reranker loaded successfully.")
        except Exception as e:
            print(f"Warning: Reranker 加载失败，将使用 RRF 分数直接排序。错误: {e}")
            self.use_rerank = False

        # 3. ChromaDB
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"找不到向量数据库路径: {db_path}")

        self.client = PersistentClient(path=db_path)
        self.collection = self.client.get_collection(
            name="chisel_knowledge",
            embedding_function=self.embedding_function,
        )
        total = self.collection.count()
        print(f"向量库就绪，共 {total} 个分块。")

        # 4. 构建 BM25 索引（一次性加载全部文档）
        print("Building BM25 index...")
        self._build_bm25_index(bm25_k1, bm25_b)
        print(f"BM25 索引构建完成，共 {len(self.all_docs)} 条文档。")

    # ----------------------------------------------------------
    # 内部：构建 BM25 索引
    # ----------------------------------------------------------
    def _build_bm25_index(self, k1: float, b: float):
        """
        从 ChromaDB 一次性读取所有文档，构建 BM25 索引。
        使用分页防止内存炸裂。
        """
        page_size = 500
        offset = 0
        all_docs = []
        all_metas = []
        all_ids = []

        while True:
            batch = self.collection.get(
                limit=page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            docs = batch.get("documents", [])
            if not docs:
                break
            all_docs.extend(docs)
            all_metas.extend(batch.get("metadatas", [{}] * len(docs)))
            all_ids.extend(batch.get("ids", [str(i + offset) for i in range(len(docs))]))
            offset += len(docs)
            if len(docs) < page_size:
                break

        self.all_docs = all_docs
        self.all_metas = all_metas
        self.all_ids = all_ids

        # 为 id → 索引位置建立映射，方便向量检索结果对齐
        self.id_to_index: dict[str, int] = {
            doc_id: idx for idx, doc_id in enumerate(self.all_ids)
        }

        self.bm25 = BM25(k1=k1, b=b)
        self.bm25.fit(self.all_docs)

    # ----------------------------------------------------------
    # 内部：source_type 过滤掩码
    # ----------------------------------------------------------
    def _get_filter_mask(self, filter_type: str | None) -> list[bool]:
        if filter_type is None:
            return [True] * len(self.all_docs)
        return [
            meta.get("source_type", "") == filter_type
            for meta in self.all_metas
        ]

    # ----------------------------------------------------------
    # 内部：RRF 融合
    # ----------------------------------------------------------
    @staticmethod
    def _rrf_merge(
        ranked_lists: list[list[int]],
        rrf_k: int,
    ) -> dict[int, float]:
        """
        输入：多个排序列表（每个列表是文档在 all_docs 中的索引，按分数从高到低）
        输出：{doc_idx: rrf_score}
        """
        rrf_scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, doc_idx in enumerate(ranked):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        return rrf_scores

    # ----------------------------------------------------------
    # 主检索接口：从规则解析蓝图中提取 query
    # ----------------------------------------------------------
    @staticmethod
    def extract_query_from_blueprint(blueprint_text: str) -> str:
        """
        从规则解析蓝图的正例部分提取检索 query。
        提取策略：
          - 代码片段中的 Chisel API 关键词（RegInit, Wire, Mux 等）
          - 代码构造逻辑自然语言描述
          - 检索关键词字段（如果蓝图包含）
        """
        parts = []

        # 1. 提取 ```scala 代码块内容
        code_blocks = re.findall(r"```scala(.*?)```", blueprint_text, re.DOTALL)
        for block in code_blocks:
            # 从代码中提取标识符（驼峰词、操作符等）
            identifiers = re.findall(r"[A-Za-z][A-Za-z0-9]*", block)
            # 过滤常见噪音词
            noise = {"val", "var", "def", "class", "object", "extends", "import",
                     "true", "false", "null", "new", "this", "when", "otherwise",
                     "io", "in", "out"}
            keywords = [w for w in identifiers if w not in noise and len(w) > 2]
            parts.extend(keywords)

        # 2. 提取自然语言描述段落
        desc_patterns = [
            r"代码构造逻辑[：:](.*?)(?:\n#|\Z)",
            r"检索描述[：:](.*?)(?:\n#|\Z)",
            r"核心修复代码[：:](.*?)```",
        ]
        for pattern in desc_patterns:
            matches = re.findall(pattern, blueprint_text, re.DOTALL)
            for m in matches:
                parts.append(m.strip())

        # 3. 提取检索关键词字段（如果规则解析器输出了这个字段）
        kw_match = re.findall(r"检索关键词[：:](.*?)(?:\n|$)", blueprint_text)
        for m in kw_match:
            parts.append(m.strip())

        query = " ".join(parts)
        # 去重同时保留顺序
        seen = set()
        deduped = []
        for word in query.split():
            if word not in seen:
                seen.add(word)
                deduped.append(word)

        return " ".join(deduped)[:512]  # 限制 query 长度

    # ----------------------------------------------------------
    # 主检索接口
    # ----------------------------------------------------------
    def search_context(
        self,
        query: str,
        filter_type: str | None = None,
        top_k: int = 3,
        initial_k: int = 15,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        score_threshold: float = 0.0,
    ) -> dict:
        """
        混合检索主入口。

        Args:
            query:          检索 query（可直接用 extract_query_from_blueprint 提取）
            filter_type:    "doc" | "code" | None
            top_k:          最终返回文档数
            initial_k:      向量检索和 BM25 各自的召回数量
            vector_weight:  向量检索在 RRF 中的权重系数（目前 RRF 本身不用权重，用于未来扩展）
            bm25_weight:    BM25 在 RRF 中的权重系数（同上）
            score_threshold: Reranker 分数阈值，低于此值的结果被过滤
        """
        if not query.strip():
            return {"context_str": "", "raw_docs": [], "hit_count": 0}

        total_docs = len(self.all_docs)
        if total_docs == 0:
            return {"context_str": "", "raw_docs": [], "hit_count": 0}

        safe_k = min(initial_k, total_docs)
        filter_mask = self._get_filter_mask(filter_type)
        valid_indices = [i for i, keep in enumerate(filter_mask) if keep]

        if not valid_indices:
            return {"context_str": "", "raw_docs": [], "hit_count": 0}

        # ── 1. 向量检索 ──────────────────────────────────────────
        chroma_filter = None
        if filter_type in ("doc", "code"):
            chroma_filter = {"source_type": {"$eq": filter_type}}

        chroma_kwargs = dict(
            query_texts=[query],
            n_results=min(safe_k, len(valid_indices)),
            include=["documents", "metadatas", "distances", "ids"],
        )
        if chroma_filter:
            chroma_kwargs["where"] = chroma_filter

        chroma_results = self.collection.query(**chroma_kwargs)

        # 向量检索结果 → 在 all_docs 中的索引列表（按相似度降序）
        vector_ranked: list[int] = []
        if chroma_results["ids"] and chroma_results["ids"][0]:
            for doc_id in chroma_results["ids"][0]:
                idx = self.id_to_index.get(doc_id)
                if idx is not None:
                    vector_ranked.append(idx)

        # ── 2. BM25 检索 ──────────────────────────────────────────
        all_bm25_scores = self.bm25.get_scores(query)

        # 只对 valid_indices 排序
        valid_bm25 = [(i, all_bm25_scores[i]) for i in valid_indices]
        valid_bm25.sort(key=lambda x: x[1], reverse=True)
        bm25_ranked: list[int] = [i for i, _ in valid_bm25[:safe_k]]

        # ── 3. RRF 融合 ───────────────────────────────────────────
        rrf_scores = self._rrf_merge(
            [vector_ranked, bm25_ranked],
            rrf_k=self.rrf_k,
        )

        # 取 RRF 分数最高的 initial_k 个候选（限定在 valid_indices 范围）
        valid_rrf = {k: v for k, v in rrf_scores.items() if k in set(valid_indices)}
        candidate_indices = sorted(valid_rrf, key=lambda i: valid_rrf[i], reverse=True)[:safe_k]

        if not candidate_indices:
            return {"context_str": "", "raw_docs": [], "hit_count": 0}

        candidate_docs = [self.all_docs[i] for i in candidate_indices]
        candidate_metas = [self.all_metas[i] for i in candidate_indices]

        # ── 4. Reranker 精排 ──────────────────────────────────────
        if self.use_rerank and len(candidate_docs) > 1:
            pairs = [[query, doc] for doc in candidate_docs]
            rerank_scores = self.reranker.predict(pairs).tolist()

            scored = [
                {
                    "content": candidate_docs[i],
                    "meta": candidate_metas[i],
                    "score": rerank_scores[i],
                    "rrf_score": valid_rrf.get(candidate_indices[i], 0.0),
                }
                for i in range(len(candidate_docs))
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)

            final_docs = [
                item for item in scored[:top_k]
                if item["score"] >= score_threshold
            ]

            if not final_docs:
                # score_threshold 过严时降级：直接取 Reranker Top-k 不过滤
                print(
                    f"Warning: 所有候选文档 Reranker 分数均低于阈值 {score_threshold}，"
                    "已降级为不过滤模式。"
                )
                final_docs = scored[:top_k]
        else:
            # 无 Reranker 时按 RRF 分数排序
            final_docs = [
                {
                    "content": candidate_docs[i],
                    "meta": candidate_metas[i],
                    "score": valid_rrf.get(candidate_indices[i], 0.0),
                    "rrf_score": valid_rrf.get(candidate_indices[i], 0.0),
                }
                for i in range(min(top_k, len(candidate_docs)))
            ]

        # ── 5. 拼接 Context ───────────────────────────────────────
        context_parts = []
        raw_docs_out = []
        current_len = 0
        max_len = 4000

        for doc in final_docs:
            content = doc["content"].strip()
            if current_len + len(content) > max_len:
                break
            context_parts.append(content)
            raw_docs_out.append(
                {
                    "source": doc["meta"].get("filename", "unknown"),
                    "source_type": doc["meta"].get("source_type", "unknown"),
                    "content": content,
                    "score": doc["score"],
                    "rrf_score": doc.get("rrf_score", 0.0),
                }
            )
            current_len += len(content)

        return {
            "context_str": "\n\n".join(context_parts),
            "raw_docs": raw_docs_out,
            "hit_count": len(raw_docs_out),
        }


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    rag = ChiselHybridRAGService()

    # ── 示例 1：直接传入 query 字符串 ────────────────────────────
    print("\n" + "=" * 60)
    print("【示例 1】直接 query 检索代码知识")
    print("=" * 60)
    res = rag.search_context(
        query="RegInit Wire Mux when io.en UInt",
        filter_type="code",
        top_k=3,
        initial_k=15,
    )
    print(f"命中 {res['hit_count']} 条文档：")
    for doc in res["raw_docs"]:
        print(
            f"  [Reranker: {doc['score']:.4f}] "
            f"[RRF: {doc['rrf_score']:.4f}] "
            f"[{doc['source_type']}] {doc['source']}"
        )
    print("\nContext 预览：")
    print(res["context_str"][:600] + ("..." if len(res["context_str"]) > 600 else ""))

    # ── 示例 2：从规则解析蓝图中自动提取 query ───────────────────
    print("\n" + "=" * 60)
    print("【示例 2】从蓝图文本自动提取 query 后检索")
    print("=" * 60)

    # 模拟规则解析阶段输出的蓝图片段（正例部分）
    blueprint_snippet = """
## 3. 正例生成蓝图 (The Positive Case)
* **核心修复代码**:
    ```scala
    val myReg = RegInit(0.U(8.W))
    when(io.en) { myReg := myReg + 1.U }
    io.out := myReg
    ```
* **代码构造逻辑**: 使用 `val` 声明不可变引用，指向 `RegInit` 时序寄存器，
  使用 `:=` 操作符进行 Chisel 信号连接，避免使用 Scala 原生赋值语句。

## 5. 知识库检索意图
* **检索关键词**: `RegInit`, `Wire`, `when`, `UInt`
* **检索描述**: RegInit 寄存器声明和 when 条件赋值的标准写法
"""

    auto_query = ChiselHybridRAGService.extract_query_from_blueprint(blueprint_snippet)
    print(f"自动提取的 Query：\n  {auto_query}\n")

    res2 = rag.search_context(
        query=auto_query,
        filter_type="code",
        top_k=3,
        initial_k=15,
    )
    print(f"命中 {res2['hit_count']} 条文档：")
    for doc in res2["raw_docs"]:
        print(
            f"  [Reranker: {doc['score']:.4f}] "
            f"[RRF: {doc['rrf_score']:.4f}] "
            f"[{doc['source_type']}] {doc['source']}"
        )
    print("\nContext 预览：")
    print(res2["context_str"][:600] + ("..." if len(res2["context_str"]) > 600 else ""))

    # ── 示例 3：检索文档知识 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("【示例 3】检索文档知识（Decoupled 握手协议）")
    print("=" * 60)
    res3 = rag.search_context(
        query="Decoupled IO valid ready handshake Queue enq deq",
        filter_type="doc",
        top_k=3,
        initial_k=15,
    )
    print(f"命中 {res3['hit_count']} 条文档：")
    for doc in res3["raw_docs"]:
        print(
            f"  [Reranker: {doc['score']:.4f}] "
            f"[RRF: {doc['rrf_score']:.4f}] "
            f"[{doc['source_type']}] {doc['source']}"
        )
    print("\nContext 预览：")
    print(res3["context_str"][:600] + ("..." if len(res3["context_str"]) > 600 else ""))