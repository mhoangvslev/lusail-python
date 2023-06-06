from io import BytesIO
from itertools import combinations
from typing import List

import pandas as pd
from SPARQLWrapper import CSV, SPARQLWrapper, JSON


def get_triple_from_edge(edge):
    s, o, p = edge
    return " ".join([s, list(p.keys())[0], o])

def get_pair_triples(triple_patterns):
    return list(combinations(triple_patterns, 2))
    #return list(permutations(triple_patterns, 2))

def exec_query_on_endpoint(query, endpoint, graph):
    # FEDSHOP
    # sparql_endpoint = SPARQLWrapper(endpoint, defaultGraph=graph)
    # sparql_endpoint.setMethod("GET")
    # sparql_endpoint.setReturnFormat(CSV)
    # sparql_endpoint.setQuery(query)
        
    # try:
    #     response = sparql_endpoint.query()
    #     result = response.convert()
    # except Exception as e:
    #     print(query)
    #     raise e
    # return response, result
    
    # LUSAIL
    result = endpoint.query(query).serialize(format="csv")    
    return None, result

def exec_query_on_relevant_sources(query: str, relavant_sources: List[str]):
    result_df = pd.DataFrame()
    for source in relavant_sources:
        # FEDSHOP
        # response, result = exec_query_on_endpoint(query, "http://localhost:34202/sparql/", source)
        
        # LUSAIL
        result = source.query(query).serialize(format="csv")
        
        # COMMON
        if len(result.decode().strip()) > 0:
            with BytesIO(result) as buffer:
                new_df = pd.read_csv(buffer)
            
            result_df = pd.concat([result_df, new_df], axis=1)
                
    return result_df