import json
import os
import uuid
from collections import defaultdict
from typing import Any, Dict, List

# import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from langchain_google_vertexai import ChatVertexAI
from prompts import GRAPH_SYSTEM_LLM_PARSER

load_dotenv()

def init_neo4j_driver():
    neo4j_uri = os.getenv("NEO4J_URI", "")
    neo4j_username = os.getenv("NEO4J_USERNAME", "")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")

    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    
    return neo4j_driver

class Triplet(BaseModel):
    h: str
    type_h: str  
    r: str       
    o: str
    type_t: str  

class KnowledgeGraph(BaseModel):
    graph: list[Triplet]

def llm_parser(prompt_input: str) -> KnowledgeGraph:
    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-001",
    )
    
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
    
    template = PromptTemplate(
        template=GRAPH_SYSTEM_LLM_PARSER,
        input_variables=["prompt_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = template | llm | parser
    
    return chain.invoke({"prompt_input": prompt_input})

def extract_graph_components(raw_data):
    parsed_response = llm_parser(raw_data)
    
    nodes = {}
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

def ingest_to_neo4j(neo4j_driver,nodes, relationships):
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

def fetch_related_graph(neo4j_driver, entity_ids):
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
    with neo4j_driver.session() as session:
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
    Supports two modes:
    - 'verbose': Provides a step-by-step reasoning (Chain-of-Thought).
    - 'concise': Provides a direct, short answer.
    """
    # Kiểm tra đầu vào hợp lệ
    if not isinstance(graph_context, dict) or "nodes" not in graph_context or "edges" not in graph_context:
        raise ValueError("graph_context must be a dict with 'nodes' and 'edges' keys.")

    # Chuyển đổi ngữ cảnh đồ thị thành chuỗi
    nodes_str = "\n- ".join(sorted(graph_context.get("nodes", [])))
    edges_str = "\n- ".join(sorted(graph_context.get("edges", [])))

    # ---- PROMPT CHO CHẾ ĐỘ NGẮN GỌN (CONCISE) ----
    GRAPH_RAG_CONCISE_PROMPT = (
        "You are a highly efficient, fact-based question-answering engine.\n"
        "Your task is to answer the user's query based **STRICTLY AND ONLY** on the provided Knowledge Graph Context.\n"
        "- Answer in a single, direct sentence.\n"
        "- Do not add any conversational fluff, explanations, or introductory phrases.\n"
        '- If the context does not contain the answer, simply state: "The provided context does not contain enough information to answer the question."'
    )
    GRAPH_RAG_VERBOSE_PROMPT = (
        "You are a highly intelligent and precise question-answering assistant.\n"
        "Your task is to answer the user's query based **STRICTLY** on the provided Knowledge Graph Context.\n"
        "Do not use any external knowledge. Follow these steps to construct your answer:\n"
        "1.  **Analyze the Query**: Briefly restate the user's main goal.\n"
        "2.  **Trace the Path**: Describe the step-by-step path you take through the graph edges to connect the key entities and find the answer. Cite the specific edges you use in parentheses, for example: `(Ultron)-[CreatedBy]->(Tony Stark)`.\n"
        "3.  **Synthesize the Final Answer**: Based on the traced path, provide a clear and concise final answer."
    )

    if mode == 'concise':
        prompt_template_str = (
            "KNOWLEDGE GRAPH CONTEXT:\n"
            "---\n"
            "Nodes:\n"
            "- {nodes_str}\n\n"
            "Edges:\n"
            "- {edges_str}\n"
            "---\n\n"
            "USER QUERY:\n"
            "\"{user_query}\"\n\n"
            "FINAL ANSWER:"
        )
        prompt_template = PromptTemplate(
            template=GRAPH_RAG_CONCISE_PROMPT + "\n" + prompt_template_str,
            input_variables=["nodes_str", "edges_str", "user_query"],
        )
    elif mode == 'verbose':
        prompt_template_str = (
            "KNOWLEDGE GRAPH CONTEXT:\n"
            "---\n"
            "Nodes:\n"
            "- {nodes_str}\n\n"
            "Edges:\n"
            "- {edges_str}\n"
            "---\n\n"
            "USER QUERY:\n"
            "\"{user_query}\"\n\n"
            "Please provide your answer following the 3-step process outlined in your instructions."
        )
        prompt_template = PromptTemplate(
            template=GRAPH_RAG_VERBOSE_PROMPT + "\n" + prompt_template_str,
            input_variables=["nodes_str", "edges_str", "user_query"],
        )
    else:
        raise ValueError("Invalid mode. Choose 'concise' or 'verbose'.")

    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-001",
    )

    chain = prompt_template | llm

    try:
        response = chain.invoke(
            {"nodes_str": nodes_str, "edges_str": edges_str, "user_query": user_query}
        )
        return response.content
    except Exception as e:
        print(f"Error querying LLM: {e}")
        from types import SimpleNamespace
        return SimpleNamespace(content=f"An error occurred: {str(e)}")


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
    
neo4j_driver = init_neo4j_driver()

print("Extracting graph components...")
nodes, relationships = extract_graph_components(raw_data)
print("Nodes:", nodes)
print("Relationships:", relationships)

print("Ingesting to Neo4j...")
node_id_mapping = ingest_to_neo4j(neo4j_driver, nodes, relationships)
print("Neo4j ingestion complete")

print("Fetching related graph...")
subgraph = fetch_related_graph(neo4j_driver, node_id_mapping.values())
print("Related graph fetched")

print("Formatting graph context...")
graph_context = format_graph_context(subgraph)
print("Graph context formatted")