import logging
import time
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.core.config import settings
from app.models.schemas import ALLOWED_NODES, ALLOWED_RELATIONSHIPS

logger = logging.getLogger(__name__)

# --- CUSTOM CYPHER GENERATION PROMPT ---
# This makes the GraphRAG significantly more robust by teaching the LLM
# how to handle case-sensitivity, messy naming, and relationship directions.
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask you to help with a Cypher statement other than this one.
Do not include any text other than the generated Cypher statement.
IMPORTANT: Output the raw Cypher query ONLY. Do NOT use markdown code blocks (```) or the word "cypher".

Critical Instructions for Robustness:
1. Always use case-insensitive matching for string properties. 
2. CHARACTERISTICS vs NAMES: If the user asks for a 'Tier-0' service, 'Legacy' DB, or 'Senior' engineer, these are likely in the `description` or `type` properties, NOT the `id`. 
   - Incorrect: `s.id CONTAINS 'Tier-0'`
   - Correct: `s.description CONTAINS 'Tier-0'` or `toLower(s.id) CONTAINS toLower('User Auth')`
3. If the user mentions a provider (AWS, GCP, Azure), check the `infrastructure` node or the `description` of the service.
4. If a query returns no results with exact matches, try using `CONTAINS` on the description property.

Relationship Directions & Constraints (STRICT):
1. (Person)-[:MANAGES]->(Person) (Use this for Manager -> Employee)
2. (Person)-[:BELONGS_TO]->(Team) (Use this for Employee -> Team)
3. (Team)-[:OWNS]->(Service) (Use this for Team -> Service)
4. (Team)-[:OWNS]->(Database)
5. (Service)-[:DEPENDS_ON]->(Service)
6. (Service)-[:CONNECTS_TO]->(Database)
7. (Service)-[:DEPLOYED_ON]->(Infrastructure)

CRITICAL RULES:
- A Person NEVER 'OWNS' a Service or Database. Only a Team does.
- A Person NEVER 'BELONGS_TO' another Person. Use 'MANAGES' instead.
- To find a Director responsible for a Service, the path is: 
  (Director:Person)-[:MANAGES*0..3]->(Engineer:Person)-[:BELONGS_TO]->(Team)-[:OWNS]->(Service)

Syntactic Rules:
- If you need to find a manager for a service: (Manager:Person)-[:MANAGES*0..3]->(Engineer:Person)-[:BELONGS_TO]->(Team)-[:OWNS]->(Service)
- If you need to check cross-cloud: (s:Service)-[:DEPLOYED_ON]->(i:Infrastructure) WHERE toLower(i.id) CONTAINS 'gcp' MATCH (s)-[:CONNECTS_TO]->(db:Database)-[:DEPLOYED_ON]->(i2:Infrastructure) WHERE toLower(i2.id) CONTAINS 'aws'
- Use simple paths first. Do not use variable length `*1..5` unless explicitly asked for cascades.

Examples:
- Manager of a Database: MATCH (p:Person)-[:MANAGES*0..2]->(p2:Person)-[:BELONGS_TO]->(t:Team)-[:OWNS]->(d:Database)
- Cross-Cloud: MATCH (s:Service)-[:DEPLOYED_ON]->(i:Infrastructure) WHERE toLower(i.id) CONTAINS 'gcp' MATCH (s)-[:CONNECTS_TO]->(db:Database) RETURN s, db

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

class GraphRAGService:
    def __init__(self):
        # 1. Connect to Neo4j
        logger.info("Connecting to Neo4j...")
        self.graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE
        )
        
        # 2. Initialize Gemini LLM
        # We use a large context window model for extraction, and high logic for Cypher
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", # Pro model is much better at complex extraction and Cypher generation
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0
        )
        
        # 3. Initialize the Graph Transformer with a flexible property approach
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=ALLOWED_NODES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            # By NOT strictly limiting node_properties, we allow Gemini to 
            # discover and extract any useful metadata it finds in the text.
            node_properties=True, 
            relationship_properties=True
        )
        
        # 4. Initialize the Cypher QA Chain
        # This translates natural language to Cypher, executes it, and answers.
        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            cypher_prompt=CYPHER_GENERATION_PROMPT, # Use our custom robust prompt
            verbose=True,
            return_intermediate_steps=True, # We want to return the actual Cypher query to the user
            allow_dangerous_requests=True   # Required for read queries in modern LangChain Neo4j wrapper
        )

    async def ingest_text(self, text: str, source: str) -> Dict[str, Any]:
        """
        Extracts entities and relationships from unstructured text and saves them to Neo4j.
        """
        start_time = time.time()
        logger.info(f"Starting Graph Extraction on text of length {len(text)}...")
        
        # Wrap text in a LangChain Document
        doc = Document(page_content=text, metadata={"source": source})
        
        # Extract Graph Documents using Gemini
        # Note: For very large texts, we would chunk this first. But Gemini 2.5 Pro 
        # has a massive context window and handles extraction best when it sees the whole context.
        graph_documents = self.llm_transformer.convert_to_graph_documents([doc])
        
        if not graph_documents:
            logger.warning("No graph entities extracted.")
            return {"nodes": 0, "edges": 0, "latency_ms": 0}
            
        logger.info(f"Extraction complete. Found {len(graph_documents[0].nodes)} nodes and {len(graph_documents[0].relationships)} edges.")
        
        # Save to Neo4j
        self.graph.add_graph_documents(
            graph_documents, 
            baseEntityLabel=True, 
            include_source=True
        )
        
        # Refresh the schema so the Cypher QA chain knows about the new data
        self.graph.refresh_schema()
        
        return {
            "nodes": len(graph_documents[0].nodes),
            "edges": len(graph_documents[0].relationships),
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

    async def query_graph(self, question: str) -> Dict[str, Any]:
        """
        Translates a natural language question into Cypher, queries Neo4j, and returns an answer.
        """
        start_time = time.time()
        logger.info(f"Generating Cypher query for: {question}")
        
        # PRO-TIP: We refresh the schema right before the query to ensure 
        # the LLM knows about any new property keys (like 'description') 
        # extracted during ingestion.
        self.graph.refresh_schema()
        
        try:
            # Execute the Cypher Chain
            result = self.cypher_chain.invoke({"query": question})
            
            answer = result.get("result", "I could not find an answer in the graph.")
            
            # Extract the raw Cypher query from the intermediate steps for educational purposes
            cypher_query = "No Cypher generated."
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps and len(intermediate_steps) > 0:
                # Usually the first intermediate step is the Cypher generation
                cypher_query = intermediate_steps[0].get("query", cypher_query)
                
            return {
                "answer": answer,
                "cypher_query": cypher_query,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
            
        except Exception as e:
            logger.error(f"Graph query failed: {str(e)}")
            return {
                "answer": f"Error querying graph: {str(e)}",
                "cypher_query": "Failed",
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }