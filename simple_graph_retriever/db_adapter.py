from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
try:
    from falkordb import FalkorDB
except ImportError:
    FalkorDB = None

from .config import logger

class GraphAdapter(ABC):
    @abstractmethod
    def query(self, cypher: str, params: dict = None) -> List[Dict[str, Any]]:
        """Executes a Cypher query and returns a list of dictionaries."""
        pass

    @abstractmethod
    def close(self):
        pass
    
    @property
    @abstractmethod
    def id_function(self) -> str:
        """Returns the Cypher function name for node IDs (e.g. 'elementId' or 'ID')."""
        pass

class Neo4jAdapter(GraphAdapter):
    def __init__(self, uri, auth):
        if GraphDatabase is None:
            raise ImportError("Neo4j client not installed. Run 'pip install simple-graph-retriever[neo4j]'")
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.driver.verify_connectivity()

    def query(self, cypher: str, params: dict = None) -> List[Dict[str, Any]]:
        if params is None:
            params = {}
        with self.driver.session() as session:
            # Neo4j's .data() automatically converts results to dicts
            return session.run(cypher, parameters=params).data()

    def close(self):
        self.driver.close()

    @property
    def id_function(self) -> str:
        return "elementId"

class FalkorDBAdapter(GraphAdapter):
    def __init__(self, host, port, username, password, graph_name):
        if FalkorDB is None:
            raise ImportError("FalkorDB client not installed. Run 'pip install falkordb'")
        
        self.client = FalkorDB(
            host=host, 
            port=port, 
            username=username, 
            password=password,
        )
        self.graph = self.client.select_graph(graph_name)
        # Ping to check connection
        self.client.connection.ping()

    def query(self, cypher: str, params: dict = None) -> List[Dict[str, Any]]:
        if params is None:
            params = {}
        
        cypher = cypher.strip().rstrip(";")
        
        try:
            result = self.graph.query(cypher, params)
        except Exception as e:
            logger.error(f"FalkorDB Query Failed: {cypher[:100]}... Error: {e}")
            raise e

        output = []
        if not result.header:
            return []
            
        # --- FIX: SMART HEADER PARSING ---
        headers = []
        for h in result.header:
            col_name = "unknown"
            
            # If h is a list/tuple like [b'id', 1] or [1, b'id']
            if isinstance(h, (list, tuple)):
                for item in h:
                    # We are looking for the Name, which is bytes or str. 
                    # We ignore the Type, which is int.
                    if isinstance(item, (bytes, str)):
                        col_name = item
                        break
            else:
                col_name = h
            
            # Decode bytes if necessary
            if isinstance(col_name, bytes):
                col_name = col_name.decode('utf-8')
                
            headers.append(str(col_name))
        # ---------------------------------

        for row in result.result_set:
            row_dict = {}
            for i, col_val in enumerate(row):
                if hasattr(col_val, 'properties'):
                    data = col_val.properties.copy()
                    data['id'] = col_val.id
                    row_dict[headers[i]] = data
                else:
                    row_dict[headers[i]] = col_val
            output.append(row_dict)
            
        return output

    def close(self):
        # Redis connections are managed by connection pool, strictly no close needed usually
        pass

    @property
    def id_function(self) -> str:
        # FalkorDB uses ID(n) which returns an Integer. 
        # We wrap it in toString later in the query construction or handle it here.
        return "ID"