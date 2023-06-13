import operator
from functools import reduce
from io import BytesIO
from itertools import chain, combinations
from typing import List, Set
import numpy as np
from rdflib import Variable
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateAlgebra, translateQuery, _traverseAgg, _addVars

import pandas as pd
from SPARQLWrapper import CSV, SPARQLWrapper, JSON

def translate_query(query_template, **kwargs):
        
    def get_variable(x, children):
        if isinstance(x, Variable):
            children.append(set([x]))
        return reduce(operator.or_, children, set())
    
    ptree = parseQuery(query_template)
    vars: set = _traverseAgg(ptree, get_variable)
    
    if kwargs.get("prefixes") is not None:
        ptree[0].extend(kwargs["prefixes"])
        
    if kwargs.get("filters") is not None:
        filters = []
        for filter in kwargs["filters"]:
            filter_vars: set = _traverseAgg(filter, get_variable)
            if len(vars.intersection(filter_vars)) > 0:
                filters.append(filter)
            
        ptree[1]["where"]["part"].extend(filters)
                    
    query_algebra = translateQuery(ptree)
    query = translateAlgebra(query_algebra)
    return query

def get_triple_from_edge(edge):
    s, o, p = edge
    return " ".join([s, list(p.keys())[0], o])

def get_pair_triples(triple_patterns):
    return list(combinations(triple_patterns, 2))
    #return list(permutations(triple_patterns, 2))

def exec_query_on_endpoint(query, endpoint, graph):
    # FEDSHOP
    sparql_endpoint = SPARQLWrapper(endpoint, defaultGraph=graph)
    sparql_endpoint.setMethod("GET")
    sparql_endpoint.setReturnFormat(CSV)
    sparql_endpoint.setQuery(query)
        
    response = sparql_endpoint.query()
    result = response.convert()
    return response, result
    
    # LUSAIL
    # result = endpoint.query(query).serialize(format="csv")    
    # return None, result

def exec_query_on_relevant_sources(query: str, relavant_sources: Set[str]):
    result_df = pd.DataFrame()
    for source in relavant_sources:
        # FEDSHOP
        response, result = exec_query_on_endpoint(query, "http://localhost:34202/sparql/", source)
        
        # LUSAIL
        # result = source.query(query).serialize(format="csv")
        
        # COMMON
        if len(result.decode().strip()) > 0:
            with BytesIO(result) as buffer:
                new_df = pd.read_csv(buffer)
            
            result_df = pd.concat([result_df, new_df], axis=0)
                
    return result_df