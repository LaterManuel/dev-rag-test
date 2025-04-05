# %%
# Type hints
from typing import Any, Dict, List, Tuple, Optional

# Standard library
import ast
import logging
import re
import warnings

# Third-party packages - Data manipulation
import pandas as pd
from tqdm import tqdm

# Third-party packages - Environment & Database
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Third-party packages - Error handling & Retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

# Langchain - Core
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Langchain - Models & Connectors
from langchain_ollama.llms import OllamaLLM


# Langchain - Graph & Experimental
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer


# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# %%
dataset = pd.read_csv('../data/dataset.csv')
dataset.head()

# %%
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
        print("Connection closed")

    def reset_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database resetted successfully!")

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]

# %%
uri = "bolt://localhost:7687"
user = "neo4j"
password = "ucsp_test"
conn = Neo4jConnection(uri, user, password)
conn.reset_database()

# %%
def parse_number(value: Any, target_type: type) -> Optional[float]:
    """Parse string to number with proper error handling."""
    if pd.isna(value):
        return None
    try:
        cleaned = str(value).strip().replace(',', '')
        return target_type(cleaned)
    except (ValueError, TypeError):
        return None

def clean_text(text: str) -> str:
    """Clean and normalize text fields."""
    if pd.isna(text):
        return ""
    return str(text).strip().title()

# %%
# Add this import at the top of your cell or with your other imports
from langchain_google_genai import GoogleGenerativeAI

# Then fix your LLM initialization
import google.generativeai as genai

# Store your API key in a variable
api_key = "AIzaSyD-r-kvXNJ_DFaVW2iG275cWgekh14o0iU"  
genai.configure(api_key=api_key)

# Initialize the LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
#llm = OllamaLLM(model="qwen2.5-coder:latest")

df = dataset.copy()

# Step 1: Define Node Labels and Properties
node_structure = "\n".join(
    [f"{col}: {', '.join(map(str, df[col].unique()[:3]))}..." for col in df.columns]
)

print(node_structure)

# %%
# Prueba simple para el LLM configurado usando predict

try:
    # Pregunta de prueba
    pregunta = "¿Cuál es la capital de Francia?"
    
    # Enviar la pregunta al modelo como un string
    respuesta = llm.predict(pregunta)
    
    # Mostrar la respuesta
    print("Respuesta del LLM:", respuesta)

except Exception as e:
    print("Error al probar el LLM:", e)

# %%
# Setup logging
from langchain.chains import LLMChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_dict_from_llm_response(response: str) -> str:
    """Extract a dictionary from an LLM response that might contain additional text or formatting."""
    # Remove any markdown code block markers
    response = re.sub(r'```(?:python|json)?|```', '', response)
    
    # Try to find content that looks like a dictionary ({...})
    dict_match = re.search(r'\{.*\}', response, re.DOTALL)
    if dict_match:
        return dict_match.group(0)
    
    return response.strip()

def validate_node_definition(node_def: Dict) -> bool:
    """Validate node definition structure"""
    if not isinstance(node_def, dict):
        return False
    return all(
        isinstance(v, dict) and all(isinstance(k, str) for k in v.keys())
        for v in node_def.values()
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_node_definitions(chain, structure: str, example: Dict) -> Dict[str, Dict[str, str]]:
    try:
        # Obtener la respuesta del LLM utilizando predict con argumentos nombrados
        response = chain.predict(structure=structure, example=example)
        
        # Preprocessing - clean up the response
        cleaned_response = extract_dict_from_llm_response(response)
        
        # Log the cleaned response for debugging
        logger.info(f"Cleaned response: {cleaned_response}")
        
        # Parsear la respuesta
        node_defs = ast.literal_eval(cleaned_response)
        
        # Validar la estructura
        if not validate_node_definition(node_defs):
            raise ValueError("Invalid node definition structure")
        return node_defs
    except (ValueError, SyntaxError) as e:
        logger.error(f"Error parsing node definitions: {e}")
        logger.error(f"Raw response: {response}")
        raise

# Updated node definition template
node_example = {
    "NodeLabel1": {"property1": "row['property1']", "property2": "row['property2'], ..."},
    "NodeLabel2": {"property1": "row['property1']", "property2": "row['property2'], ..."},
    "NodeLabel3": {"property1": "row['property1']", "property2": "row['property2'], ..."},
}

define_nodes_prompt = PromptTemplate(
    input_variables=["example", "structure"],
    template=("""
        Analyze the dataset structure below and extract the entity labels for nodes and their properties.
        The node properties should be based on the dataset columns and their values.
        
        Return ONLY a valid Python dictionary like this format, with NO additional text:
        {example}
        
        Dataset Structure:
        {structure}
        
        Do not include any explanation, markdown formatting, or code block indicators.
        Your response must be ONLY the Python dictionary that can be parsed with ast.literal_eval().
        """)
)

# Execute with error handling
try:
    node_chain = LLMChain(llm=llm, prompt=define_nodes_prompt)

    node_definitions = get_node_definitions(node_chain, structure=node_structure, example=node_example)
    logger.info(f"Node Definitions: {node_definitions}")
except Exception as e:
    logger.error(f"Failed to get node definitions: {e}")
    raise

# %%
class RelationshipIdentifier:
    """Identifies relationships between nodes in a graph database."""
    
    RELATIONSHIP_EXAMPLE = [
        ("NodeLabel1", "RelationshipLabel", "NodeLabel2"),
        ("NodeLabel1", "RelationshipLabel", "NodeLabel3"),
        ("NodeLabel2", "RelationshipLabel", "NodeLabel3"),
    ]


    PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["structure", "node_definitions", "example"],
    template="""
        Consider the following Dataset Structure:\n{structure}\n\n

        Consider the following Node Definitions:\n{node_definitions}\n\n

        Based on the dataset structure and node definitions, identify relationships (edges) between nodes.\n
        Return the relationships as a list of triples where each triple contains the start node label, relationship label, and end node label, and each triple is a tuple.\n
        Please return only the list of tuples. Please do not report triple backticks to identify a code block, just return the list of tuples.\n\n

        Example:\n{example}
        """
)

    def __init__(self, llm: Any, logger: logging.Logger = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.chain = self.PROMPT_TEMPLATE | self.llm

    def validate_relationships(self, relationships: List[Tuple]) -> bool:
        """Validate relationship structure."""
        return all(
            isinstance(rel, tuple) and 
            len(rel) == 3 and 
            all(isinstance(x, str) for x in rel)
            for rel in relationships
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def identify_relationships(self, structure: str, node_definitions: Dict) -> List[Tuple]:
        """Identify relationships with retry logic."""
        try:
            # response = self.chain.run(
            #     structure=structure,
            #     node_definitions=str(node_definitions),
            #     example=str(self.RELATIONSHIP_EXAMPLE)
            # )
            response = self.chain.invoke({
                "structure": structure, 
                "node_definitions": str(node_definitions), 
                "example": str(self.RELATIONSHIP_EXAMPLE)
            })
            
            relationships = ast.literal_eval(response)
            
            if not self.validate_relationships(relationships):
                raise ValueError("Invalid relationship structure")
                
            self.logger.info(f"Identified {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error identifying relationships: {e}")
            raise

    def get_relationship_types(self) -> List[str]:
        """Extract unique relationship types."""
        return list(set(rel[1] for rel in self.identify_relationships()))

# Usage
identifier = RelationshipIdentifier(llm=llm)
relationships = identifier.identify_relationships(node_structure, node_definitions)
print("Relationships:", relationships)

# %%
class CypherQueryBuilder:
    """Builds Cypher queries for Neo4j graph database."""

    INPUT_EXAMPLE = """
    NodeLabel1: value1, value2
    NodeLabel2: value1, value2
    """
    
    EXAMPLE_CYPHER = example_cypher = """
    CREATE (n1:NodeLabel1 {property1: "row['property1']", property2: "row['property2']"})
    CREATE (n2:NodeLabel2 {property1: "row['property1']", property2: "row['property2']"})
    CREATE (n1)-[:RelationshipLabel]->(n2);
    """

    PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["structure", "node_definitions", "relationships", "example"],
    template="""
        Consider the following Node Definitions:\n{node_definitions}\n\n
        Consider the following Relationships:\n{relationships}\n\n
        Generate Cypher queries to create nodes and relationships using the node definitions and relationships below. Remember to replace the placeholder values with actual data from the dataset.\n
        Include all the properties in the Node Definitions for each node as defined and create relationships.\n
        Return a single string with each query separated by a semicolon.\n
        Don't include any other text or quotation marks in the response.\n
        Please return only the string containing Cypher queries. Please do not report triple backticks to identify a code block.\n\n

        Example Input:\n{input}\n\n

        Example Output Cypher query:\n{cypher}
    """
)

    def __init__(self, llm: Any, logger: logging.Logger = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        # self.chain = LLMChain(llm=llm, prompt=self.PROMPT_TEMPLATE)
        self.chain = self.PROMPT_TEMPLATE | self.llm

    def validate_cypher_query(self, query: str) -> bool:
        """Validate Cypher query syntax using LLM and regex patterns."""
        
        VALIDATION_PROMPT = PromptTemplate(
            input_variables=["query"],
            template="""
            Validate this Cypher query and return TRUE or FALSE:
            
            Query: {query}
            
            Rules to check:
            1. Valid CREATE statements
            2. Proper property formatting
            3. Valid relationship syntax
            4. No missing parentheses
            5. Valid property names
            6. Valid relationship types
            
            Return only TRUE if query is valid, FALSE if invalid.
            """
        )
        
        try:
            # Basic pattern validation
            basic_valid = all(re.search(pattern, query) for pattern in [
                r'CREATE \(',  
                r'\{.*?\}',    
                r'\)-\[:.*?\]->'
            ])
            
            if not basic_valid:
                return False
                
            # LLM validation
            validation_chain = VALIDATION_PROMPT | self.llm
            result = validation_chain.invoke({"query": query})
            
            # Parse result
            is_valid = "TRUE" in result.upper()
            
            if not is_valid:
                self.logger.warning(f"LLM validation failed for query: {query}")
                
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def sanitize_query(self, query: str) -> str:
        """Sanitize and format Cypher query."""
        return (query
                .strip()
                .replace('\n', ' ')
                .replace('  ', ' ')
                .replace("'row[", "row['")
                .replace("]'", "']"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def build_queries(self, node_definitions: Dict, relationships: List) -> str:
        """Build Cypher queries with retry logic."""
        try:
            response = self.chain.invoke({
                "node_definitions": str(node_definitions),
                "relationships": str(relationships),
                "input": self.INPUT_EXAMPLE,
                "cypher": self.EXAMPLE_CYPHER
            })

            # Get response inside triple backticks
            if '```' in response:
                response = response.split('```')[1]

            
            # Sanitize response
            queries = self.sanitize_query(response)
            
            # Validate queries
            if not self.validate_cypher_query(queries):
                raise ValueError("Invalid Cypher query syntax")
                
            self.logger.info("Successfully generated Cypher queries")
            return queries
            
        except Exception as e:
            self.logger.error(f"Error building Cypher queries: {e}")
            raise

    def split_queries(self, queries: str) -> List[str]:
        """Split combined queries into individual statements."""
        return [q.strip() for q in queries.split(';') if q.strip()]

# Usage
builder = CypherQueryBuilder(llm=llm)
cypher_queries = builder.build_queries(node_definitions, relationships)
print("Cypher Queries:", cypher_queries)

# %%
# Iterate over dataframe with progress bar
logs = ""
total_rows = len(df)

def sanitize_value(value):
    """Properly sanitize and format values for Cypher queries."""
    if pd.isna(value):
        return "null"
    if isinstance(value, str):
        # Escape double quotes inside the string and wrap with quotes
        escaped_value = value.replace('"', '\\"').replace("'", "\\'")
        return f'"{escaped_value}"'
    if isinstance(value, (int, float)):
        return str(value)
    # For other types, convert to string and wrap in quotes
    return f'"{str(value)}"'

for index, row in tqdm(df.iterrows(), 
                      total=total_rows,
                      desc="Loading data to Neo4j",
                      position=0,
                      leave=True):
    
    # Replace placeholders with actual values
    cypher_query = cypher_queries
    for column in df.columns:
        placeholder = f"row['{column}']"
        if placeholder in cypher_query:
            cypher_query = cypher_query.replace(
                placeholder, 
                sanitize_value(row[column])
            )
    
    try:
        # Execute query and update progress
        conn.execute_query(cypher_query)
    except Exception as e:
        logs += f"Error on row {index+1}: {str(ae)}\n"
        print(f"Error on row {index+1}: {str(e)[:200]}...")  # Print truncated error for immediate feedback

# Display logs
print(logs) # Uncomment to display logs

# %%
query = """
MATCH p=(pub:Publication)-[r]-(related)
RETURN p
LIMIT 5;
"""
conn.execute_query(query)


# %%
query = """
CALL db.labels()
"""
conn.execute_query(query)



