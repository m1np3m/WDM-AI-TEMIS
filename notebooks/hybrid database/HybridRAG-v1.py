from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from collections import defaultdict
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
import uuid
import os
import json
import pandas as pd
import os
import google.generativeai as genai
import json 

load_dotenv()
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
openai_key = os.getenv("OPENAI_API_KEY")

neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

collection_name = "graphRAGstoreds"
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

try:
    qdrant_client.delete_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' has been deleted successfully.")
except Exception as e:
    print(f"Could not delete collection (it might not exist, which is OK): {e}")

# class single(BaseModel):
#     node: str
#     target_node: str
#     relationship: str

# class GraphComponents(BaseModel):
#     graph: list[single]

from pydantic import BaseModel

# Lớp Pydantic mới, tổng quát hơn
class Triplet(BaseModel):
    h: str
    type_h: str  
    r: str       
    o: str
    type_t: str  

class KnowledgeGraph(BaseModel):
    graph: list[Triplet]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openai_key,
)

def openai_llm_parser(prompt_input: str) -> KnowledgeGraph:
    """
    Extracts structured information using a generalized prompt, 
    allowing the LLM to infer entity types and relationships.
    """
    

    system_prompt = f"""Your task is to act as an expert information extractor. From the provided INPUT_TEXT, you will extract a knowledge graph.

    The output must be a JSON object with a single key "graph", which contains a list of structured objects. Each object represents a relationship triplet and must have the following keys: 'h', 'type_h', 'r', 'o', 'type_t'.

    GUIDELINES:
    1.  'h' (head) and 'o' (tail) are the entities.
    2.  'type_h' and 'type_t' are the general categories. You must infer these types. Types should be concise, capitalized, singular nouns (e.g., PERSON, COMPANY, VEHICLE, LOCATION, PRODUCT).
    3.  **Crucially, identify abstract concepts like EVENTS (e.g., 'Battle of New York', 'Ultron's Attack') and PROTOCOLS (e.g., 'Sokovia Accords').**
    4.  'r' (relationship) is a short, active verb.
      - For actions between entities, use verbs like: Drove, Invented, Created, Wields, Led, Defeated.
      - **For cause-and-effect, use verbs like: Caused, LedTo, ResultedIn.**
      - **For participation, use: ParticipatedIn.**
    5.  **Entity Disambiguation**: Consolidate different names for the same entity.
    6.  **Simplicity**: Keep entity names short and specific.

    EXAMPLE 1 (Business):
    - Input: 'The 2008 financial crisis led to the creation of the Dodd-Frank Act.'
    - Output:
    {{
      "graph": [
        {{ "h": "2008 Financial Crisis", "type_h": "EVENT", "r": "LedTo", "o": "Dodd-Frank Act", "type_t": "PROTOCOL" }}
      ]
    }}

    EXAMPLE 2 (MCU - a more relevant example for you):
    - Input: 'The Battle of New York was a major conflict where the Avengers first assembled to fight Loki.'
    - Output:
    {{
      "graph": [
          {{ "h": "Avengers", "type_h": "GROUP", "r": "ParticipatedIn", "o": "Battle of New York", "type_t": "EVENT" }},
          {{ "h": "Loki", "type_h": "PERSON", "r": "ParticipatedIn", "o": "Battle of New York", "type_t": "EVENT" }}
      ]
    }}

    Your output MUST be a valid JSON object. Do not add any text before or after the JSON.

    ===========================================================
    INPUT_TEXT:
    {prompt_input}
    """
    
    completion = client.chat.completions.create(
        model="qwen/qwen3-235b-a22b:free",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt
            }
        ]
    )
    
    response_content = completion.choices[0].message.content
    print("Raw response from model:", response_content)

    try:
        return KnowledgeGraph.model_validate_json(response_content)
    except Exception as e:
        print(f"Pydantic validation failed: {e}")
        return KnowledgeGraph(graph=[])
    

def extract_graph_components(raw_data):
    # Hàm openai_llm_parser bây giờ nhận trực tiếp raw_data
    parsed_response = openai_llm_parser(raw_data)
    
    nodes = {}  # Lưu trữ {node_name: {"id": uuid, "type": entity_type}}
    relationships = []

    for triplet in parsed_response.graph:
        # Trích xuất thông tin từ triplet
        head_name, head_type = triplet.h, triplet.type_h
        tail_name, tail_type = triplet.o, triplet.type_t
        rel_type = triplet.r

        # Thêm nút đầu (head) nếu chưa tồn tại
        if head_name not in nodes:
            nodes[head_name] = {"id": str(uuid.uuid4()), "type": head_type}

        # Thêm nút cuối (tail) nếu chưa tồn tại
        if tail_name not in nodes:
            nodes[tail_name] = {"id": str(uuid.uuid4()), "type": tail_type}

        # Thêm mối quan hệ
        relationships.append({
            "source": nodes[head_name]["id"],
            "target": nodes[tail_name]["id"],
            "type": rel_type
        })

    return nodes, relationships


def ingest_to_neo4j(nodes, relationships):
    with neo4j_driver.session() as session:
        # Tạo các nút với label (type) động
        for name, properties in nodes.items():
            # Sử dụng apoc.create.node để đặt label động
            # Cần đảm bảo Neo4j có cài đặt APOC
            query = f"""
            CREATE (n:{properties['type']} {{id: $id, name: $name}})
            """
            session.run(
                query,
                id=properties['id'],
                name=name
            )

        # Tạo mối quan hệ (cạnh)
        for rel in relationships:
            # Sử dụng f-string để tạo loại quan hệ động
            query = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            CREATE (a)-[:{rel['type']}]->(b)
            """
            session.run(
                query,
                source_id=rel["source"],
                target_id=rel["target"]
            )
    # Trả về mapping từ tên sang ID để Qdrant sử dụng
    node_id_mapping = {name: props["id"] for name, props in nodes.items()}
    return node_id_mapping


def create_collection(client, collection_name, vector_dimension):
    # Try to fetch the collection status
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        # If collection does not exist, an error will be thrown, so we create the collection
        if 'Not found: Collection' in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")



api_key = os.getenv("GEMINI_API_KEY")
    
genai.configure(api_key=api_key)

def gemini_embeddings(text: str, task: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Tạo vector embedding cho một đoạn văn bản sử dụng mô hình của Gemini.

    Args:
        text (str): Đoạn văn bản cần embedding.
        task (str): Loại tác vụ cho embedding. Các giá trị phổ biến bao gồm:
                    - "RETRIEVAL_QUERY": Cho truy vấn tìm kiếm.
                    - "RETRIEVAL_DOCUMENT": Cho tài liệu trong cơ sở dữ liệu vector.
                    - "SEMANTIC_SIMILARITY": Cho việc so sánh sự tương đồng ngữ nghĩa.
                    - "CLASSIFICATION": Cho các tác vụ phân loại.
                    - "CLUSTERING": Cho các tác vụ phân cụm.
    
    Returns:
        list[float]: Một danh sách các số thực biểu diễn vector embedding.
    """
    model = "models/embedding-001"
    
    try:
        response = genai.embed_content(
            model=model,
            content=text,
            task_type=task
        )
        
        return response['embedding']
        
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tạo embedding: {e}")
        return []
    


def ingest_to_qdrant(collection_name, node_id_mapping):
    """
    Creates embeddings for the names of the extracted nodes and ingests them into Qdrant.
    This approach is more robust and logical than embedding raw text lines.
    """
    points_to_upsert = []
    
    print(f"Preparing to create embeddings for {len(node_id_mapping)} nodes...")
    
    # Lặp qua các node đã được trích xuất
    for node_name, node_id in node_id_mapping.items():
        if not node_name or not node_name.strip():
            print(f"Skipping empty node name for id {node_id}")
            continue

        # Tạo embedding cho tên của node
        embedding = gemini_embeddings(node_name, task="RETRIEVAL_DOCUMENT")
        
        if embedding:
            points_to_upsert.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),      # ID cho điểm vector trong Qdrant
                    vector=embedding,
                    payload={"id": node_id}    # ID của node trong Neo4j để liên kết
                )
            )
        else:
            print(f"Failed to create embedding for node: {node_name}")

    # Upsert tất cả các điểm hợp lệ cùng một lúc
    if points_to_upsert:
        print(f"Ingesting {len(points_to_upsert)} points into Qdrant...")
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points_to_upsert
        )
    else:
        print("No valid points to ingest into Qdrant.")


def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    results = retriever.search(query_vector=gemini_embeddings(query, task="RETRIEVAL_QUERY"), top_k=5)
    
    return results



def fetch_related_graph(neo4j_client, entity_ids):
    """
    Fetches a subgraph around the given entity IDs using a more robust
    variable-length path expansion.
    This query finds all paths up to 3 hops away from the starting entities
    and collects all unique nodes and relationships along these paths.
    """
    query = """
    MATCH path = (e)-[*1..3]-(related_node)
    WHERE e.id IN $entity_ids
    UNWIND nodes(path) AS n
    UNWIND relationships(path) AS r
    RETURN collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS relationships
    """
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids).single()
        if not result or not result['nodes']:
            return []
        
        nodes = result['nodes']
        relationships = result['relationships']
        
        subgraph = []
        # Tạo một map từ element_id của nút sang đối tượng nút để tra cứu dễ dàng
        node_map = {node.element_id: node for node in nodes}

        for rel in relationships:
            # Đảm bảo cả nút đầu và nút cuối của mối quan hệ đều nằm trong map đã thu thập
            if rel.start_node.element_id in node_map and rel.end_node.element_id in node_map:
                subgraph.append({
                    "entity": rel.start_node,
                    "relationship": rel,
                    "related_node": rel.end_node
                })
    return subgraph


def format_graph_context(subgraph):
    """
    Formats the subgraph into a text representation for the LLM.
    This version is more robust and handles potential missing properties.
    """
    nodes = set()
    edges = set() # Sử dụng set để tự động loại bỏ các cạnh trùng lặp

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        # Sử dụng .get() để truy cập an toàn, đề phòng thiếu thuộc tính
        entity_name = entity.get("name", "Unknown Entity")
        related_name = related.get("name", "Unknown Entity")
        
        # loại của mối quan hệ là `type` của đối tượng relationship
        rel_type = relationship.type 

        nodes.add(entity_name)
        nodes.add(related_name)
        edges.add(f"({entity_name})-[{rel_type}]->({related_name})")

    return {"nodes": sorted(list(nodes)), "edges": sorted(list(edges))}

def graphRAG_run(graph_context, user_query, mode='concise'):
    """
    Runs the final generation step of the GraphRAG pipeline.
    This version supports two modes:
    - 'verbose': Provides a step-by-step reasoning (Chain-of-Thought).
    - 'concise': Provides a direct, short answer.
    """
    
    # Chuyển đổi ngữ cảnh đồ thị thành chuỗi
    nodes_str = "\n- ".join(sorted(graph_context["nodes"]))
    edges_str = "\n- ".join(sorted(graph_context["edges"]))

    # ---- PROMPT CHO CHẾ ĐỘ NGẮN GỌN (CONCISE) ----
    if mode == 'concise':
        system_prompt = """You are a highly efficient, fact-based question-answering engine. 
        Your task is to answer the user's query based **STRICTLY AND ONLY** on the provided Knowledge Graph Context.
        - Answer in a single, direct sentence.
        - Do not add any conversational fluff, explanations, or introductory phrases.
        - If the context does not contain the answer, simply state: "The provided context does not contain enough information to answer the question."
        """
        user_prompt = f"""
        KNOWLEDGE GRAPH CONTEXT:
        ---
        Nodes:
        - {nodes_str}

        Edges:
        - {edges_str}
        ---

        USER QUERY:
        "{user_query}"

        FINAL ANSWER:
        """
    elif mode == 'verbose':
        system_prompt = """You are a highly intelligent and precise question-answering assistant. 
        Your task is to answer the user's query based **STRICTLY** on the provided Knowledge Graph Context. 
        Do not use any external knowledge. Follow these steps to construct your answer:
        1.  **Analyze the Query**: Briefly restate the user's main goal.
        2.  **Trace the Path**: Describe the step-by-step path you take through the graph edges to connect the key entities and find the answer. Cite the specific edges you use in parentheses, for example: `(Ultron)-[CreatedBy]->(Tony Stark)`.
        3.  **Synthesize the Final Answer**: Based on the traced path, provide a clear and concise final answer.
        """
        user_prompt = f"""
        KNOWLEDGE GRAPH CONTEXT:
        ---
        Nodes:
        - {nodes_str}

        Edges:
        - {edges_str}
        ---

        USER QUERY:
        "{user_query}"

        Please provide your answer following the 3-step process outlined in your instructions.
        """
    else:
        raise ValueError("Invalid mode. Choose 'concise' or 'verbose'.")

    try:
        response = client.chat.completions.create(
            model='mistralai/mistral-small-3.1-24b-instruct:free',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error querying LLM: {e}")
        from types import SimpleNamespace
        return SimpleNamespace(content=f"An error occurred: {str(e)}")
    

if __name__ == "__main__":
    print("Script started")
    print("Loading environment variables...")
    load_dotenv('.env.local')
    print("Environment variables loaded")
    
    print("Initializing clients...")
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key
    )
    print("Clients initialized")
    
    print("Creating collection...")
    collection_name = "graphRAGstoreds"
    vector_dimension = 768
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified")
    
    print("Extracting graph components...")
    
    raw_data = """
    # Marvel Cinematic Universe (MCU) Knowledge Base in Tables

    ### Table 1: Characters and Identities
    | Character Name      | Alias / Superhero Name | Affiliation | Key Detail                               |
    |---------------------|------------------------|-------------|------------------------------------------|
    | Tony Stark          | Iron Man               | Avengers    | A genius inventor and founding member    |
    | Steve Rogers        | Captain America        | Avengers    | A super-soldier from World War II        |
    | Thor Odinson        | God of Thunder         | Avengers    | Hails from the realm of Asgard           |
    | Peter Parker        | Spider-Man             | None        | Mentored by Tony Stark                   |
    | Wanda Maximoff      | Scarlet Witch          | Avengers    | Possesses reality-warping abilities      |
    | T'Challa            | Black Panther          | None        | King of the nation of Wakanda            |
    | Thanos              | The Mad Titan          | None        | A warlord from the planet Titan          |
    | Howard Stark        | -                      | -           | Father of Tony Stark, an inventor        |
    | Loki                | God of Mischief        | -           | Brother of Thor, led Chitauri army       |
    | Ultron              | -                      | -           | An artificial intelligence created by Tony Stark |
    | Vision              | -                      | Avengers    | An android created by Ultron and Tony Stark |


    ### Table 2: Artifacts, Materials, and Their Connections
    | Artifact / Material | Type               | Connection / Owner / Source           | Notes                                           |
    |---------------------|--------------------|---------------------------------------|-------------------------------------------------|
    | Infinity Stones     | Artifact Set       | Sought by Thanos                      | Six powerful singularities                      |
    | Tesseract           | Infinity Stone     | Used by Loki, formerly held by HYDRA  | Contains the Space Stone                        |
    | Mind Stone          | Infinity Stone     | Used to create Vision                 | Was embedded in Vision's forehead               |
    | Arc Reactor         | Power Source       | Powers Iron Man's suit                | Technology invented by Howard Stark             |
    | Captain's Shield    | Weapon             | Wielded by Captain America            | Made of the rare metal Vibranium                |
    | Vibranium           | Material           | Found only in Wakanda                 | The primary resource of Wakanda                 |
    | Mjolnir             | Weapon             | Wielded by Thor                       | A mystical hammer forged in Nidavellir          |


    ### Table 3: Events and Consequences
    | Event Name            | Location       | Key Participants       | Outcome / Consequence                                 |
    |-----------------------|----------------|------------------------|-------------------------------------------------------|
    | Battle of New York    | New York City  | Avengers, Loki, Chitauri | The Avengers first assembled to defeat Loki's army.   |
    | Ultron's Attack       | Sokovia        | Avengers, Ultron       | Major destruction in the city of Sokovia.             |
    | Sokovia Accords       | -              | Global Governments     | A legal protocol introduced to regulate superheroes.  |
    | Infinity War          | Multiple       | Avengers, Thanos       | Thanos successfully collected all Infinity Stones.    |
    """

    nodes, relationships = extract_graph_components(raw_data)
    print("Nodes:", nodes)
    print("Relationships:", relationships)
    
    print("Ingesting to Neo4j...")
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Neo4j ingestion complete")
    
    
    print("Ingesting to Qdrant...")
    ingest_to_qdrant(collection_name, node_id_mapping)
    print("Qdrant ingestion complete")

    query = "Who created Ultron?"
    print("Starting retriever search...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Retriever results:", retriever_result)
    
    print("Extracting entity IDs...")
    entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
    print("Entity IDs:", entity_ids)
    
    print("Fetching related graph...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Subgraph:", subgraph)
    
    print("Formatting graph context...")
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)
    
    print("Running GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)