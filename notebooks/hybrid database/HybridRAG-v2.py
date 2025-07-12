# -*- coding: utf-8 -*-
"""
HybriRAG v2 - Script Chạy Pipeline RAG Lai Ghép Vector và Đồ Thị.

Mô tả:
    Script này triển khai một hệ thống RAG (Retrieval-Augmented Generation) lai ghép,
    kết hợp sức mạnh của cả cơ sở dữ liệu vector (ChromaDB) và cơ sở dữ liệu
    đồ thị (Neo4j) để trả lời các câu hỏi phức tạp.

    Quy trình hoạt động chính:
    1.  **Phân rã câu hỏi (Query Transformation)**: Một LLM được sử dụng để phân tích
        câu hỏi của người dùng, tách nó thành các câu hỏi con hướng đến tìm kiếm
        vector (dữ liệu mô tả) và các thực thể chính hướng đến tìm kiếm đồ thị
        (dữ liệu thực tế, có cấu trúc).
    2.  **Truy vấn song song (Parallel Retrieval)**: Hệ thống đồng thời truy vấn
        dữ liệu từ ChromaDB (dựa trên các câu hỏi con) và Neo4j (dựa trên các
        thực thể).
    3.  **Tăng cường tín hiệu đồ thị (Graph Signal Boosting)**: Các kết quả từ
        tìm kiếm vector được "tăng điểm" nếu chúng chứa các thực thể được tìm thấy
        hoặc có liên quan trong đồ thị tri thức.
    4.  **Sắp xếp lại (Reranking)**: Một mô hình reranker được áp dụng để đánh giá
        lại và sắp xếp các kết quả đã được kết hợp và tăng cường.
    5.  **Tổng hợp câu trả lời (Response Synthesis)**: LLM cuối cùng sẽ tổng hợp
        câu trả lời mạch lạc dựa trên những thông tin đã được chọn lọc và xếp hạng.

Yêu cầu thư viện:
    - llama-index-graph-stores-neo4j
    - llama-index-vector-stores-chroma
    - llama-index-embeddings-huggingface
    - llama-index-llms-openrouter
    - python-dotenv
    - sentence-transformers
    - chromadb
    - neo4j

Cài đặt:
    pip install llama-index-graph-stores-neo4j llama-index-vector-stores-chroma \\
                llama-index-embeddings-huggingface llama-index-llms-openrouter \\
                python-dotenv sentence-transformers

Thiết lập môi trường:
    Tạo một file `.env` trong cùng thư mục với script và điền các thông tin sau:
    OPENAI_API_KEY="sk-or-v1-..." (Đây là key của OpenRouter)
    YOUR_SITE_URL="http://localhost:8888"
    YOUR_SITE_NAME="VectorGraphRAG PoC"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="your_neo4j_password"
    NEO4J_URI="bolt://localhost:7687"

Cách chạy:
    python HybriRAG_v2_script.py
"""

import os
import re
import json
import logging
import asyncio
from typing import List, Optional, Dict, Any

import chromadb
from dotenv import load_dotenv

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    get_response_synthesizer,
    QueryBundle
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.llms import ChatMessage
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter


# --- CẤU HÌNH BAN ĐẦU ---

# Thiết lập logging để theo dõi quá trình hoạt động
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


def configure_global_settings():
    """
    Cấu hình các thiết lập toàn cục cho LlamaIndex (LLM và model embedding).
    """
    Settings.llm = OpenRouter(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="mistralai/mistral-7b-instruct:free",
        default_headers={
            "HTTP-Referer": os.getenv("YOUR_SITE_URL", "http://localhost:8888"),
            "X-Title": os.getenv("YOUR_SITE_NAME", "VectorGraphRAG PoC"),
        },
        max_tokens=512,
        temperature=0.1,
    )
    logging.info(f"✅ LLM đã được cấu hình để sử dụng model '{Settings.llm.model}' từ OpenRouter.")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )
    logging.info(f"✅ Embedding model '{Settings.embed_model.model_name}' đã được tải.")


def setup_datastores() -> tuple[Neo4jGraphStore, VectorStoreIndex]:
    """
    Khởi tạo và nạp dữ liệu vào Neo4j và ChromaDB.

    Hàm này sẽ xóa dữ liệu cũ trong cả hai cơ sở dữ liệu trước khi nạp dữ liệu mới.
    Đây là một bước thiết lập, thường chỉ cần chạy một lần.

    Returns:
        Một tuple chứa đối tượng graph_store của Neo4j và vector_index của ChromaDB.
    """
    # --- Dữ liệu mẫu ---
    text_docs = [
        Document(
            text="Triết lý thiết kế của Steve Jobs tại Apple luôn tập trung vào sự đơn giản, trực quan và đặt trải nghiệm người dùng làm trung tâm. Ông tin rằng sản phẩm không chỉ cần mạnh mẽ về chức năng mà còn phải là một tác phẩm nghệ thuật, đẹp từ trong ra ngoài. Điều này đã định hình nên các sản phẩm biểu tượng như iPhone và MacBook.",
            metadata={"source": "Phân tích nội bộ"}
        ),
        Document(
            text="iPhone 15, ra mắt năm 2023, tiếp tục kế thừa di sản thiết kế của Apple với khung viền titan, cổng USB-C và hệ thống camera cải tiến mạnh mẽ. Nó được xem là một trong những smartphone hàng đầu thị trường, thể hiện rõ triết lý của công ty.",
            metadata={"source": "Đánh giá sản phẩm TechReview"}
        ),
        Document(
            text="Cuộc đối đầu giữa Apple và Samsung là một trong những câu chuyện kinh điển của làng công nghệ. Trong khi Apple tập trung vào hệ sinh thái khép kín và sự tối ưu hóa phần cứng-phần mềm, Samsung lại đa dạng hóa sản phẩm với nhiều phân khúc và tiên phong trong các công nghệ màn hình gập.",
            metadata={"source": "Bài báo kinh doanh"}
        ),
    ]
    graph_triplets = [
        ("Steve Jobs", "FOUNDED", "Apple"),
        ("Apple", "PRODUCES", "iPhone 15"),
        ("Apple", "HAS_PHILOSOPHY", "Thiết kế tối giản và trải nghiệm người dùng"),
        ("Samsung", "PRODUCES", "Galaxy S24"),
        ("Samsung", "COMPETES_WITH", "Apple")
    ]

    # --- INGESTION VÀO NEO4J (GraphDB) ---
    logging.info("--- Bắt đầu Ingest vào Neo4j ---")
    graph_store = Neo4jGraphStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        database="neo4j",
    )
    # Xóa toàn bộ dữ liệu cũ để đảm bảo sự sạch sẽ
    graph_store.query("MATCH (n) DETACH DELETE n")
    for subj, pred, obj in graph_triplets:
        graph_store.upsert_triplet(subj, pred, obj)
    logging.info("--- Ingest vào Neo4j Hoàn tất ---")

    # --- INGESTION VÀO CHROMADB (VectorDB) ---
    logging.info("--- Bắt đầu Ingest vào ChromaDB ---")
    db = chromadb.PersistentClient(path="./chroma_db")
    # Xóa collection cũ nếu tồn tại
    try:
        db.delete_collection("vector_graph_rag_poc")
        logging.info("Collection 'vector_graph_rag_poc' cũ đã được xóa.")
    except ValueError:
        logging.info("Không tìm thấy collection cũ để xóa, sẽ tạo mới.")
        
    chroma_collection = db.get_or_create_collection("vector_graph_rag_poc")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_index = VectorStoreIndex.from_documents(
        text_docs, vector_store=vector_store
    )
    logging.info("--- Ingest vào ChromaDB Hoàn tất ---")

    return graph_store, vector_index


# --- CÁC THÀNH PHẦN CỦA PIPELINE RAG ---

def transform_query(query: str) -> dict:
    """
    Sử dụng LLM để phân rã câu hỏi phức tạp thành các thành phần con.

    Hàm này gửi câu hỏi gốc đến LLM với một prompt được thiết kế đặc biệt để
    yêu cầu LLM trả về một cấu trúc JSON. Cấu trúc này bao gồm:
    - 'vector_search_queries': Danh sách các câu hỏi con, đầy đủ ngữ nghĩa,
      phù hợp để tìm kiếm thông tin mô tả trong cơ sở dữ liệu vector.
    - 'graph_entities': Danh sách các thực thể (danh từ riêng, khái niệm) chính
      có trong câu hỏi, dùng để truy vấn các mối quan hệ trong đồ thị tri thức.

    Args:
        query: Câu hỏi gốc từ người dùng.

    Returns:
        Một dictionary chứa 'vector_search_queries' và 'graph_entities'.
        Nếu LLM trả về JSON không hợp lệ, hàm sẽ fallback bằng cách sử dụng
        câu hỏi gốc làm truy vấn vector duy nhất và danh sách thực thể rỗng.
    """
    prompt = f"""
Bạn là một trợ lý AI hữu ích, chuyên phân tích và phân rã các câu hỏi phức tạp của người dùng thành các nhiệm vụ con cho một hệ thống RAG. Mục tiêu của bạn là tách câu hỏi gốc thành một danh sách các câu hỏi con độc lập và trích xuất các thực thể chính. Một số câu hỏi con sẽ được dùng để truy vấn cơ sở dữ liệu vector (để lấy thông tin mô tả), và các thực thể sẽ được dùng để truy vấn đồ thị tri thức (để lấy dữ liệu thực tế và các mối quan hệ).

Dựa trên câu hỏi của người dùng, hãy tạo một đối tượng JSON chứa hai danh sách: 'vector_search_queries' và 'graph_entities'.

- 'vector_search_queries': Chứa các câu hỏi mô tả, đầy đủ ý nghĩa.
- 'graph_entities': Chứa các thực thể chính được xác định trong câu hỏi.

User Question: "{query}"

{{
  "vector_search_queries": [],
  "graph_entities": []
}}
"""
    response = Settings.llm.chat([ChatMessage(role="user", content=prompt)])
    content = response.message.content

    try:
        # Cố gắng trích xuất JSON nếu nó nằm trong khối markdown code
        json_match = re.search(r"```json\n(.*?)\n```", content, re.S)
        json_str = json_match.group(1) if json_match else content
        
        result = json.loads(json_str)
        # Đảm bảo các khóa cần thiết tồn tại
        if 'vector_search_queries' not in result:
            result['vector_search_queries'] = []
        if 'graph_entities' not in result:
            result['graph_entities'] = []
            
        return result
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        logging.error(f"Lỗi khi phân tích JSON từ LLM: {e}")
        logging.error(f"Nội dung nhận được từ LLM: '{content}'")
        logging.info("Sử dụng fallback: trả về câu hỏi gốc làm truy vấn vector.")
        return {"vector_search_queries": [query], "graph_entities": []}


def get_connected_entities(graph_store: Neo4jGraphStore, entity: str) -> List[str]:
    """
    Truy vấn Neo4j để tìm các thực thể có kết nối trực tiếp với một thực thể cho trước.

    Args:
        graph_store: Đối tượng kết nối đến Neo4j.
        entity: Tên của thực thể cần tìm các mối quan hệ.

    Returns:
        Một danh sách các ID của các thực thể được kết nối.
    """
    query = f"""
    MATCH (e1 {{id: "{entity}"}})-- (e2)
    RETURN e2.id
    """
    try:
        result = graph_store.query(query)
        return [item['e2.id'] for item in result] if result else []
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn get_connected_entities cho '{entity}': {e}")
        return []


class GraphSignalBooster(BaseNodePostprocessor):
    """
    Một bộ xử lý hậu kỳ (Node Postprocessor) để tăng điểm cho các node văn bản
    dựa trên tín hiệu từ đồ thị tri thức.

    Lớp này hoạt động bằng cách:
    1. Nhận một danh sách các thực thể chính từ câu hỏi (`graph_entities`).
    2. Mở rộng danh sách này bằng cách truy vấn đồ thị để tìm các thực thể
       liên quan trực tiếp (1-hop neighbors).
    3. Khi xử lý các node văn bản được truy xuất từ VectorDB, nó sẽ kiểm tra
       xem văn bản của node có chứa bất kỳ thực thể nào trong danh sách
       mở rộng không.
    4. Nếu có, điểm (score) của node đó sẽ được nhân với một hệ số (`weight`),
       giúp "đẩy" các node có liên quan đến đồ thị lên vị trí cao hơn trong
       danh sách kết quả cuối cùng.
    
    Attributes:
        graph_entities (List[str]): Danh sách các thực thể ban đầu từ câu hỏi.
        weight (float): Hệ số để tăng điểm cho các node phù hợp.
    """
    graph_store: Neo4jGraphStore
    graph_entities: List[str] = []
    weight: float = 1.5
    _related_entities: set = set()

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        graph_entities: List[str],
        weight: float = 1.5,
        **kwargs: Any,
    ):
        """Khởi tạo GraphSignalBooster."""
        super().__init__(
            graph_store=graph_store,
            graph_entities=graph_entities, 
            weight=weight, 
            **kwargs
        )
        self._related_entities = set(self.graph_entities)
        for entity in self.graph_entities:
            connected = get_connected_entities(self.graph_store, entity)
            self._related_entities.update(connected)
        logging.info(f"[GraphSignalBooster] Các thực thể liên quan từ đồ thị: {self._related_entities}")

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """Tăng điểm cho các node chứa thực thể liên quan."""
        boosted_nodes = []
        for node_with_score in nodes:
            node_text = node_with_score.node.get_content().lower()
            boost_factor = 1.0
            found_entities = []
            
            for entity in self._related_entities:
                # Tìm kiếm thực thể dưới dạng một từ hoàn chỉnh
                if re.search(r'\b' + re.escape(entity.lower()) + r'\b', node_text):
                    boost_factor = self.weight
                    found_entities.append(entity)
            
            if boost_factor > 1.0:
                logging.info(
                    f"[GraphSignalBooster] Tăng điểm cho node (source: "
                    f"{node_with_score.node.metadata.get('source')}) vì chứa "
                    f"các thực thể liên quan: {found_entities}"
                )
                node_with_score.score *= boost_factor

            boosted_nodes.append(node_with_score)
        
        # Sắp xếp lại danh sách node sau khi đã tăng điểm
        boosted_nodes.sort(key=lambda x: x.score, reverse=True)
        return boosted_nodes


async def retrieve_from_vector_db(retriever: VectorIndexRetriever, queries: List[str]) -> List[NodeWithScore]:
    """
    Thực hiện truy vấn bất đồng bộ đến cơ sở dữ liệu vector với nhiều câu hỏi.

    Args:
        retriever: Đối tượng retriever của VectorStoreIndex.
        queries: Danh sách các câu hỏi con để truy vấn.

    Returns:
        Một danh sách phẳng chứa tất cả các NodeWithScore được tìm thấy.
    """
    tasks = [retriever.aretrieve(q) for q in queries]
    nested_results = await asyncio.gather(*tasks)
    
    all_nodes = []
    # Loại bỏ các node trùng lặp dựa trên ID
    seen_node_ids = set()
    for result_list in nested_results:
        for node in result_list:
            if node.node.node_id not in seen_node_ids:
                all_nodes.append(node)
                seen_node_ids.add(node.node.node_id)

    logging.info(f"Tổng cộng truy xuất được {len(all_nodes)} node duy nhất từ VectorDB.")
    return all_nodes


async def retrieve_from_graph_db(graph_store: Neo4jGraphStore, entities: List[str]) -> List[Document]:
    """
    Thực hiện truy vấn bất đồng bộ đến cơ sở dữ liệu đồ thị cho một danh sách các thực thể.

    Args:
        graph_store: Đối tượng kết nối đến Neo4j.
        entities: Danh sách các thực thể để truy vấn.

    Returns:
        Một danh sách các đối tượng Document, mỗi Document chứa thông tin
        về các mối quan hệ của một thực thể.
    """
    async def query_entity(entity):
        # Sửa câu truy vấn để lấy cả 3 thành phần của triplet
        cypher_query = f"MATCH (n {{id: '{entity}'}})-[r]-(m) RETURN n.id, type(r) as rel_type, m.id"
        try:
            # Chạy truy vấn blocking trong một aio thread để không chặn event loop
            graph_results = await asyncio.to_thread(graph_store.query, cypher_query)
            
            if graph_results:
                logging.info(f"  > Kết quả từ GraphDB cho '{entity}': Đã tìm thấy {len(graph_results)} mối quan hệ.")
                # Sửa cách truy cập kết quả cho phù hợp với câu query mới
                text_result = f"Thông tin từ đồ thị về '{entity}':\n" + "\n".join(
                    [f"- {res['n.id']} {res['rel_type']} {res['m.id']}" for res in graph_results]
                )
                return Document(text=text_result, metadata={"source": "GraphDB"})
            return None
        except Exception as e:
            logging.error(f"  > Lỗi khi truy vấn GraphDB cho '{entity}': {e}")
            return None

    tasks = [query_entity(entity) for entity in entities]
    results = await asyncio.gather(*tasks)
    # Lọc ra những kết quả không phải None
    return [doc for doc in results if doc is not None]


async def run_hybrid_rag_pipeline(
    user_query: str, 
    graph_store: Neo4jGraphStore, 
    vector_index: VectorStoreIndex
):
    """
    Thực thi toàn bộ pipeline RAG lai ghép.

    Args:
        user_query: Câu hỏi đầu vào từ người dùng.
        graph_store: Đối tượng graph_store đã được khởi tạo.
        vector_index: Đối tượng vector_index đã được khởi tạo.
    """
    print("\n" + "="*50)
    print(f"🚀 Bắt đầu xử lý câu hỏi: '{user_query}'")
    print("="*50 + "\n")

    # 1. Query Transformation
    print("--- Bước 1: Phân rã câu hỏi ---")
    transformed = transform_query(user_query)
    vector_queries = transformed.get('vector_search_queries', [user_query])
    graph_entities = transformed.get('graph_entities', [])
    print(f"  - Vector Queries: {vector_queries}")
    print(f"  - Graph Entities: {graph_entities}\n")
    
    # 2. Parallel Retrieval
    print("--- Bước 2: Truy vấn song song từ VectorDB và GraphDB ---")
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    
    vector_task = retrieve_from_vector_db(vector_retriever, vector_queries)
    graph_task = retrieve_from_graph_db(graph_store, graph_entities)
    
    vector_retrieved_nodes, graph_retrieved_docs = await asyncio.gather(vector_task, graph_task)
    print("  - Truy vấn song song hoàn tất.\n")

    # 3. Combine and Rerank
    print("--- Bước 3: Kết hợp, Tăng cường và Sắp xếp lại kết quả ---")
    # Kết hợp kết quả từ hai nguồn
    combined_nodes_and_docs = vector_retrieved_nodes + graph_retrieved_docs

    # Khởi tạo các bộ xử lý hậu kỳ
    graph_booster = GraphSignalBooster(
        graph_store=graph_store, 
        graph_entities=graph_entities, 
        weight=1.8 # Tăng trọng số để tín hiệu đồ thị rõ ràng hơn
    )
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3", 
        top_n=5 # Chỉ giữ lại 5 kết quả tốt nhất sau khi rerank
    )

    # Áp dụng các bộ xử lý
    # Lưu ý: LlamaIndex áp dụng theo thứ tự trong danh sách
    # Ở đây ta sẽ rerank trước, sau đó mới boost tín hiệu từ graph
    # Điều này đảm bảo các tài liệu liên quan nhất được chọn, sau đó ưu tiên những cái có thực thể đồ thị.
    reranked_nodes = reranker.postprocess_nodes(combined_nodes_and_docs, query_bundle=QueryBundle(user_query))
    logging.info(f"[Reranker] Đã giảm số lượng node xuống còn {len(reranked_nodes)}.")
    
    final_nodes_for_synthesis = graph_booster.postprocess_nodes(reranked_nodes, query_bundle=QueryBundle(user_query))
    logging.info("[GraphSignalBooster] Đã áp dụng tăng cường tín hiệu đồ thị.")
    
    print("  - Kết hợp và xử lý hậu kỳ hoàn tất.\n")

    # 4. Response Synthesis
    print("--- Bước 4: Tổng hợp câu trả lời cuối cùng ---")
    response_synthesizer = get_response_synthesizer(response_mode="compact")
    
    final_response = await response_synthesizer.asynthesize(
        query_bundle=QueryBundle(user_query),
        nodes=final_nodes_for_synthesis,
    )
    
    print("\n" + "="*50)
    print("✅ Câu trả lời cuối cùng:")
    print("="*50)
    print(str(final_response))
    print("\n--- Các nguồn đã sử dụng để tạo câu trả lời: ---")
    for node in final_response.source_nodes:
        source = node.metadata.get('source', 'N/A')
        content_preview = node.get_content().strip().replace('\n', ' ')[:150]
        print(f"- Điểm: {node.score:.4f} | Nguồn: {source}")
        print(f"  Nội dung: {content_preview}...")
    print("="*50 + "\n")


async def main():
    """
    Hàm chính để thiết lập và chạy pipeline.
    """
    configure_global_settings()
    graph_store, vector_index = setup_datastores()
    
    # Câu hỏi ví dụ
    complex_query = "Phân tích triết lý thiết kế của Steve Jobs và nó ảnh hưởng đến iPhone 15 như thế nào, một sản phẩm cạnh tranh với Samsung?"
    
    await run_hybrid_rag_pipeline(complex_query, graph_store, vector_index)


if __name__ == "__main__":
    # Chạy hàm main bất đồng bộ
    asyncio.run(main())