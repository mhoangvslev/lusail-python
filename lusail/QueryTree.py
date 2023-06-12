from io import BytesIO
from itertools import product
from typing import List, Union
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import ParseResults
from rdflib import Graph, Literal, URIRef, Variable
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.namespace import RDF, RDFS
import networkx as nx

from lusail.Subquery import Subquery
from lusail.utils import exec_query_on_endpoint, translate_query

LUSAIL_CACHE = {}

def load_lusail_example():
    g1 = Graph()
    
    # Entities    
    CMU = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#CMU")
    MIT = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#MIT")
    Lee = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#Lee")
    OS = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#OS")
    Meg = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#Meg")
    Ann = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#Ann")
    Ben = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#Ben")
    
    # Relations
    
    g1.add((OS, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#graduateCourse")))
    
    g1.add((MIT, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#University")))
    g1.add((MIT, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#address"), Literal("XXX")))

    g1.add((CMU, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#University")))
    
    g1.add((Lee, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#graduateStudent")))
    g1.add((Lee, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#MScDegreeFrom"), CMU))
    g1.add((Lee, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#takesCourse"), OS))
    g1.add((Lee, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#advisor"), Ben))
    
    g1.add((Meg, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#graduateStudent")))
    g1.add((Meg, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#takesCourse"), OS))
    g1.add((Meg, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#advisor"), Ann))

    g1.add((Ann, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#associateProfessor")))    
    g1.add((Ann, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#PhDDegreeFrom"), MIT))
    g1.add((Ann, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#works"), MIT))

    g1.add((Ben, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#associateProfessor")))
    g1.add((Ben, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#PhDDegreeFrom"), MIT))
    g1.add((Ben, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#works"), MIT))
    g1.add((Ben, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#address"), Literal("ZZZ")))
    g1.add((Ben, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#teacherOf"), OS))
    
    
    g2 = Graph()
    
    DB = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#DB")
    Tim = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#Tim")
    Kim = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#Kim")
    DM = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#DB")
    Joy = URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#Joy")
    
    g2.add((DB, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#graduateCourse")))
    g2.add((DM, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#graduateCourse")))
    
    g2.add((MIT, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#University")))
    
    g2.add((CMU, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#University")))
    g2.add((CMU, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#address"), Literal("CCC")))
    
    g2.add((Tim, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#associateProfessor")))
    g2.add((Tim, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#PhDDegreeFrom"), MIT))
    g2.add((Tim, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#works"), CMU))
    g2.add((Tim, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#teacherOf"), DB))

    g2.add((Joy, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#associateProfessor")))    
    g2.add((Joy, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#teacherOf"), DM))
    g2.add((Joy, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#PhDDegreeFrom"), CMU))
    
    g2.add((Kim, RDF.type, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#graduateStudent")))
    g2.add((Kim, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#takesCourse"), DB))
    g2.add((Kim, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#takesCourse"), DM))
    g2.add((Kim, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#advisor"), Tim))
    g2.add((Kim, URIRef("http://swat.cse.lehigh.edu/onto/univ-bench.owl#advisor"), Joy))
    
    return [g1, g2]

EP1, EP2 = load_lusail_example()

class QueryTree:
    def __init__(self, query_txt: str):
        
        self._hyperGraph = nx.MultiDiGraph()

        self._query = query_txt
        self._prefixes = []
        self._bgps = []
        self._filters = []
        self._triple_patterns = {}
        self._type_assertions = {}
        self._orderby = None
        self._limitoffset = None

        # Parse query
        self._parse_tree = parseQuery(self._query)     
        self.parseTree(self._parse_tree)
        
        # nx.draw(self.hyperGraph, with_labels=True)
        # plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    def parseTree(self, node):
        if isinstance(node, ParseResults) or isinstance(node, list):
            return [ self.parseTree(item) for item in node ]
        elif isinstance(node, CompValue):
            if node.name == "PrefixDecl":
                # prefix = node["prefix"]
                # iri = node["iri"]
                # self.prefixes[prefix] = iri
                self._prefixes.append(node)
            elif node.name == "SelectQuery":
                if "modifier" in node.keys():
                    self.modifier = node["modifier"]
                
                if "projection" in node.keys():
                    self._projection = self.parseTree(node["projection"])
                where = self.parseTree(node["where"])
                
                if "orderby" in node.keys():
                    self._orderby = node["orderby"]
                
                if "limitoffset" in node.keys():
                    self._limitoffset = node["limitoffset"]
                
            elif node.name == "GroupGraphPatternSub":
                self._bgps.append(self.parseTree(node["part"]))
            elif node.name == "GroupOrUnionGraphPattern":
                self._bgps.extend(self.parseTree(node["graph"]))
            elif node.name == "SubSelect":
                return node
            elif node.name == "TriplesBlock":
                # Do not parse triples
                # Construct hyperGraph
                for triple in node["triples"]:
                    s, p, o = triple
                    sdata = self.parseTree(s)
                    pdata = np.array(self.parseTree(p)).item()
                    odata = self.parseTree(o)
                    
                    self._hyperGraph.add_node(sdata)
                    self._hyperGraph.add_node(odata)
                    edge_data = {pdata: 1}
                    self._hyperGraph.add_edge(sdata, odata, **edge_data)
                    
                    key = " ".join((sdata, pdata, odata))
                    self._triple_patterns[key] = triple
                    
                    if pdata == "rdf:type":
                        self._type_assertions[sdata] = key
                    
            elif node.name == "PathAlternative":
                return self.parseTree(node["part"])
            elif node.name == "PathElt":
                return self.parseTree(node["part"])
            elif node.name == "PathSequence":
                return self.parseTree(node["part"])
            elif node.name == "pname":
                # prefix = self.prefixes[node['prefix']]
                # return f"<{prefix}{node['localname']}>"
                return f"{node['prefix']}:{node['localname']}"
            elif node.name == "vars":
                return node["var"].n3()  
            elif node.name == "Filter":
                self._filters.append(node)  
            else:
                print(node)
                raise RuntimeError(f"CompValue of type {node.name} is not yet supported")
        elif isinstance(node, Variable) or isinstance(node, URIRef) or isinstance(node, Literal):
            return node.n3()
        else:
            print(node)
            raise RuntimeError(f"Type {type(node)} not yet supported")
    
    def get_relevant_sources(self, tripleOrSubquery: Union[str, Subquery]) -> List[str]:
        
        if tripleOrSubquery in LUSAIL_CACHE.keys():
            return LUSAIL_CACHE[tripleOrSubquery]
                
        parse_tree = []
        parse_tree.append(self._prefixes)
        
        query = None
        
        if isinstance(tripleOrSubquery, Subquery):
            query = tripleOrSubquery.stringify(isAskQuery=True, prefixes=self._prefixes)
        else:
            query_template = f"""SELECT * WHERE {{
                {tripleOrSubquery} .
            }} LIMIT 1
            """
                        
            query = translate_query(query_template, prefixes=self._prefixes, filters=self._filters)
                    
        relevant_sources = []
        
        # FEDSHOP
        with open("test/fedshop/graphs.txt", "r") as efs:
            graphs = efs.read().splitlines()
        
            for graph in graphs:
                # params = {"default-graph-uri": graph}
                # endpoint = f"http://localhost:34202/sparql/?default-graph-uri={urlencode(params)}"
                # print(endpoint)
                response, result = exec_query_on_endpoint(query, "http://localhost:34202/sparql/", graph)
                with BytesIO(result) as buffer:
                    result_df = pd.read_csv(buffer)
                    if not result_df.empty:
                        relevant_sources.append(graph)
        
        # LUSAIL
        # endpoints = [EP1, EP2]
        # for endpoint in endpoints:
        #     _, result = exec_query_on_endpoint(query, endpoint, None)
        #     with BytesIO(result) as buffer:
        #         result_df = pd.read_csv(buffer)
        #         if not result_df.empty:
        #             relevant_sources.append(endpoint)
        
        LUSAIL_CACHE[tripleOrSubquery] = relevant_sources
        return relevant_sources 
        
    def get_triple_patterns(self) -> List[str]:
        triples = list(self._triple_patterns.keys())
        triples = [ triple for triple in triples if triple not in self._type_assertions.values() ]
        return triples
    
    def get_variables(self) -> set:
        return [ node for node in list(self._hyperGraph.nodes) if str(node).startswith("?") ]
    
    def get_edges(self, vrtx) -> set:
        # Order in which edges are exploited affect the decomposition outcome 
        edges = list(self._hyperGraph.in_edges(vrtx, data=True)) + list(self._hyperGraph.out_edges(vrtx, data=True))
        edges = [ (s, o, p) for (s, o, p) in edges if list(p.keys())[0] != "rdf:type" ]
        return edges
    
    def bind_subqueries(self, subqueries: List[Subquery]):
        
        # Build SERVICE template, VALUES        
        subQ_templates = []
        subQ_vars = []
        subQ_relS = []
        for subQID, subQ in enumerate(subqueries):
            
            subQ_vars.append(f"?sq{subQID}")
            
            subQ_templates.append(
            f"""SERVICE ?sq{subQID} {{
                { ' . '.join(subQ.get_triple_patterns())}
            }}
            """)
            
            subQ_relS.append([ URIRef(relS).n3() for relS in self.get_relevant_sources(subQ)])
            
        relS_values_combinations = [ f"( {' '.join(comb)} )" for comb in product(*subQ_relS) ]
        
        # Build final query template
        query_template = f"""SELECT {' '.join(self._projection)} WHERE {{
            VALUES { ' '.join(subQ_vars) } {{ {' '.join(relS_values_combinations)} }}
            { ' . '.join(subQ_templates) }
        }}
        """
        
        query = translate_query(query_template, prefixes=self._prefixes, filters=self._filters)
        print(query)
        return query    
    
    def __repr__(self) -> str:
        return self._query

    def __str__(self) -> str:
        return str(self._parse_tree)

# if __name__ == "__main__":
#     with open("test/injected.sparql", "r") as qfs:
#         algebra = QueryTree(qfs.read())