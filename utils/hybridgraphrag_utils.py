from neo4j import GraphDatabase
from qdrant_client import QdrantClient
import os
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever

class GraphRAG:
    def __init__(self):
        self.qdrant_key = os.getenv("QDRANT_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)

        from .neo4j_qdrant_utils import (
            extract_graph_components,
            openai_llm_parser,
            ingest_to_neo4j,
            create_collection,
            ingest_to_qdrant,
            retriever_search,
            fetch_related_graph,
            format_graph_context,
            graphRAG_run
        )
        self.openai_llm_parser = openai_llm_parser
        self.extract_graph_components = extract_graph_components
        self.ingest_to_neo4j = ingest_to_neo4j
        self.create_collection = create_collection
        self.ingest_to_qdrant = ingest_to_qdrant
        self.retriever_search = retriever_search
        self.fetch_related_graph = fetch_related_graph
        self.format_graph_context = format_graph_context
        self.graphRAG_run = graphRAG_run

    def add_documents(self, collection_name, raw_data, vector_dimension=768):
        self.create_collection(self.qdrant_client, collection_name, vector_dimension)
        nodes, relationships = self.extract_graph_components(raw_data)
        node_id_mapping = self.ingest_to_neo4j(nodes, relationships)
        self.ingest_to_qdrant(collection_name, node_id_mapping)

    def get_documents(self, collection_name, query):
        results = self.retriever_search(
            self.neo4j_driver,
            self.qdrant_client,
            collection_name,
            query
        )
        return results

    def full_pipeline(self, collection_name, query, mode='concise'):
        results = self.get_documents(collection_name, query)
        entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in results.items]
        subgraph = self.fetch_related_graph(self.neo4j_driver, entity_ids)
        graph_context = self.format_graph_context(subgraph)
        return self.graphRAG_run(graph_context, query, mode=mode)
