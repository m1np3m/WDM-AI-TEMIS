import json
import os
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from langchain_google_vertexai import ChatVertexAI
from langfuse import Langfuse

from .rag import RAG
from .prompts import GRAPH_SYSTEM_LLM_PARSER
from .setting import K

load_dotenv()

class Triplet(BaseModel):
    h: str
    type_h: str  
    r: str       
    o: str
    type_t: str  

class KnowledgeGraph(BaseModel):
    graph: list[Triplet]

class HybridRAG(RAG):
    """
    Hybrid RAG class that combines traditional vector search with Neo4j graph database
    """
    
    def __init__(
        self,
        embedding_type: str,
        embedding_model: str,
        enable_hybrid_search: bool,
        chunk_type: str,
        use_memory: bool,
        collection_name: str,
        persist_dir: str,
        use_reranker: bool,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        langfuse_client: Optional[Langfuse] = None,
    ):
        # Initialize parent RAG class
        super().__init__(
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            enable_hybrid_search=enable_hybrid_search,
            chunk_type=chunk_type,
            use_memory=use_memory,
            collection_name=collection_name,
            persist_dir=persist_dir,
            use_reranker=use_reranker,
            langfuse_client=langfuse_client,
        )
        
        # Initialize Neo4j connection
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            logger.warning("Neo4j credentials not provided. Graph features will be disabled.")
            self.neo4j_driver = None
        else:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_username, self.neo4j_password)
            )
            logger.info("Neo4j connection established")
        
        # Store entity mappings for hybrid retrieval
        self.entity_mappings = {}

    def _get_qdrant_client(self) -> QdrantClient:
        """Extract QdrantClient from VectorStore"""
        return self.vectorstore._get_client()

    def _llm_parser(self, prompt_input: str) -> KnowledgeGraph:
        """Extract knowledge graph using LangChain ChatVertexAI"""
        llm = ChatVertexAI(
            model_name="gemini-2.0-flash-001",
            temperature=0.1,
        )
        
        parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        
        template = PromptTemplate(
            template=GRAPH_SYSTEM_LLM_PARSER,
            input_variables=["prompt_input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = template | llm | parser
        
        return chain.invoke({"prompt_input": prompt_input})

    def _extract_graph_components(self, raw_data: str):
        """Extract nodes and relationships from raw data"""
        parsed_response = self._llm_parser(raw_data)
        
        nodes = {}
        relationships = []

        for triplet in parsed_response.graph:
            head_name, head_type = triplet.h, triplet.type_h
            tail_name, tail_type = triplet.o, triplet.type_t
            rel_type = triplet.r

            # Add head node if not exists
            if head_name not in nodes:
                nodes[head_name] = {"id": str(uuid.uuid4()), "type": head_type}

            # Add tail node if not exists
            if tail_name not in nodes:
                nodes[tail_name] = {"id": str(uuid.uuid4()), "type": tail_type}

            # Add relationship
            relationships.append({
                "source": nodes[head_name]["id"],
                "target": nodes[tail_name]["id"],
                "type": rel_type
            })

        return nodes, relationships

    def _ingest_to_neo4j(self, nodes: dict, relationships: list):
        """Ingest nodes and relationships to Neo4j"""
        if not self.neo4j_driver:
            logger.warning("Neo4j not available. Skipping graph ingestion.")
            return {}
            
        with self.neo4j_driver.session() as session:
            # Create nodes with dynamic labels
            for name, properties in nodes.items():
                query = f"""
                CREATE (n:{properties['type']} {{id: $id, name: $name}})
                """
                session.run(
                    query,
                    id=properties['id'],
                    name=name
                )

            # Create relationships
            for rel in relationships:
                query = f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                CREATE (a)-[:{rel['type']}]->(b)
                """
                session.run(
                    query,
                    source_id=rel["source"],
                    target_id=rel["target"]
                )
        
        # Return mapping for Qdrant integration
        node_id_mapping = {name: props["id"] for name, props in nodes.items()}
        return node_id_mapping

    def _ingest_entities_to_qdrant(self, node_id_mapping: dict):
        """Ingest entity embeddings to Qdrant using VectorStore embeddings"""
        if not node_id_mapping:
            return
            
        # Create documents for entities
        entity_docs = []
        for entity_name, entity_id in node_id_mapping.items():
            doc = Document(
                page_content=entity_name,
                metadata={
                    "type": "entity",
                    "entity_id": entity_id,
                    "source": "knowledge_graph"
                }
            )
            entity_docs.append(doc)
        
        # Add to vectorstore
        self.vectorstore.add_documents(entity_docs)
        logger.info(f"Added {len(entity_docs)} entity embeddings to vector store")

    def _fetch_related_graph(self, entity_ids: List[str]):
        """Fetch subgraph around given entity IDs"""
        if not self.neo4j_driver:
            return []
            
        query = """
        MATCH path = (e)-[*1..3]-(related_node)
        WHERE e.id IN $entity_ids
        UNWIND nodes(path) AS n
        UNWIND relationships(path) AS r
        RETURN collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS relationships
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(query, entity_ids=entity_ids).single()
            if not result or not result['nodes']:
                return []
            
            nodes = result['nodes']
            relationships = result['relationships']
            
            subgraph = []
            node_map = {node.element_id: node for node in nodes}

            for rel in relationships:
                if rel.start_node.element_id in node_map and rel.end_node.element_id in node_map:
                    subgraph.append({
                        "entity": rel.start_node,
                        "relationship": rel,
                        "related_node": rel.end_node
                    })
        return subgraph

    def _format_graph_context(self, subgraph: List[dict]):
        """Format subgraph for LLM consumption"""
        nodes = set()
        edges = set()

        for entry in subgraph:
            entity = entry["entity"]
            related = entry["related_node"]
            relationship = entry["relationship"]

            entity_name = entity.get("name", "Unknown Entity")
            related_name = related.get("name", "Unknown Entity")
            rel_type = relationship.type 

            nodes.add(entity_name)
            nodes.add(related_name)
            edges.add(f"({entity_name})-[{rel_type}]->({related_name})")

        return {"nodes": sorted(list(nodes)), "edges": sorted(list(edges))}

    def _hybrid_retrieve(self, query: str, top_k: int = K):
        """Retrieve entities using vector similarity and fetch related graph"""
        if not self.neo4j_driver:
            logger.warning("Neo4j not available. Falling back to regular retrieval.")
            return self.retrieve_documents(query)
        
        # Get entity documents from vector search
        entity_docs = self.vectorstore.retrieve_documents(
            query=query,
            filter_types=["entity"],
            num_docs=top_k
        )
        
        if not entity_docs:
            logger.warning("No entities found. Falling back to regular retrieval.")
            return self.retrieve_documents(query)
        
        # Extract entity IDs
        entity_ids = []
        for doc in entity_docs:
            entity_id = doc.metadata.get("entity_id")
            if entity_id:
                entity_ids.append(entity_id)
        
        if not entity_ids:
            return self.retrieve_documents(query)
        
        # Fetch related subgraph
        subgraph = self._fetch_related_graph(entity_ids)
        
        return subgraph

    def _graph_generate_response(self, query: str, graph_context: dict, mode='concise') -> str:
        """Generate response using graph context"""
        nodes_str = "\n- ".join(sorted(graph_context.get("nodes", [])))
        edges_str = "\n- ".join(sorted(graph_context.get("edges", [])))

        if mode == 'concise':
            system_prompt = (
                "You are a highly efficient, fact-based question-answering engine.\n"
                "Your task is to answer the user's query based **STRICTLY AND ONLY** on the provided Knowledge Graph Context.\n"
                "- Answer in a single, direct sentence.\n"
                "- Do not add any conversational fluff, explanations, or introductory phrases.\n"
                '- If the context does not contain the answer, simply state: "The provided context does not contain enough information to answer the question."'
            )
            
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
                template=system_prompt + "\n" + prompt_template_str,
                input_variables=["nodes_str", "edges_str", "user_query"],
            )
        else:
            raise ValueError("Only 'concise' mode is supported for now")

        llm = ChatVertexAI(
            model_name="gemini-2.0-flash-001",
            temperature=0.0,
        )

        chain = prompt_template | llm

        try:
            response = chain.invoke(
                {"nodes_str": nodes_str, "edges_str": edges_str, "user_query": query}
            )
            return response.content
        except Exception as e:
            logger.error(f"Error in graph generation: {e}")
            return f"An error occurred: {str(e)}"

    def ingest_knowledge_graph(self, raw_data: str):
        """Main method to ingest raw data as knowledge graph"""
        if not self.neo4j_driver:
            logger.warning("Neo4j not available. Cannot ingest knowledge graph.")
            return
            
        logger.info("Extracting graph components...")
        nodes, relationships = self._extract_graph_components(raw_data)
        
        logger.info(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
        
        logger.info("Ingesting to Neo4j...")
        node_id_mapping = self._ingest_to_neo4j(nodes, relationships)
        
        logger.info("Ingesting entities to Qdrant...")
        self._ingest_entities_to_qdrant(node_id_mapping)
        
        # Store mappings for later use
        self.entity_mappings.update(node_id_mapping)
        
        logger.info("Knowledge graph ingestion completed!")

    def hybrid_query(self, query: str, use_graph: bool = True, mode: str = 'concise') -> dict:
        """
        Enhanced query method that can use both vector search and graph traversal
        
        Args:
            query: User query
            use_graph: Whether to use graph-based retrieval
            mode: Generation mode ('concise' or 'verbose')
        
        Returns:
            Dictionary with response and metadata
        """
        if not use_graph or not self.neo4j_driver:
            # Fallback to regular RAG
            return super().__call__(query)
        
        # Hybrid retrieval: entities + graph traversal
        subgraph = self._hybrid_retrieve(query)
        
        if not subgraph:
            logger.info("No graph context found. Using regular RAG.")
            return super().__call__(query)
        
        # Format graph context
        graph_context = self._format_graph_context(subgraph)
        
        # Generate response using graph context
        graph_response = self._graph_generate_response(query, graph_context, mode)
        
        # Also get regular RAG response for comparison
        regular_result = super().__call__(query)
        
        return {
            "response": graph_response,
            "graph_context": graph_context,
            "regular_response": regular_result["response"],
            "regular_docs": regular_result["docs"],
            "query": query,
            "mode": "hybrid_graph",
            "subgraph_size": len(subgraph)
        }

    def __call__(self, query: str, use_graph: bool = False, **kwargs) -> dict:
        """
        Main query interface with optional graph usage
        
        Args:
            query: User query  
            use_graph: Whether to use hybrid graph retrieval
            **kwargs: Additional arguments
        """
        if use_graph:
            return self.hybrid_query(query, use_graph=True, **kwargs)
        else:
            return super().__call__(query, **kwargs)

    def close(self):
        """Clean up connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed") 