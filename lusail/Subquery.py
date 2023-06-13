from collections import Counter
from copy import deepcopy
from typing import Set
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateAlgebra, translateQuery

from lusail.utils import translate_query

class Subquery:
    def __init__(self, edge, parent) -> None:        
        self.__parent = parent
        self.__children = []
        s, _, o = str(edge).split()
        
        if parent == s: 
            if o.startswith("?"):
                self.__children.append(o)
        elif parent == o :
            if s.startswith("?"):
                self.__children.append(s)
        else: raise ValueError(f"{parent} is neither subject or object!")
        
        self.__edges = [edge]
        
    @staticmethod
    def merge(*subqueries):
        sq = deepcopy(subqueries[0])
        for subQ in subqueries[1:]:
            sq.update(subQ.get_triple_patterns())
        return sq   
            
    def get_parent(self):
        return self.__parent
    
    def get_children(self):
        return self.__children
    
    def get_triple_patterns(self):
        return self.__edges
    
    def get_join_variables(self) -> Set[str]:
        # return set([self.__parent] + self.__children)
        qvars = []
        for triple in self.__edges:
            s, _, o = triple.split()
            if s.startswith("?"): qvars.append(s)
            if o.startswith("?"): qvars.append(o)
            
        qvars_count = Counter(qvars)
        jvs = [ qvar for qvar, count in qvars_count.items() if count > 1 ]        
        return set(jvs)
    
    def get_variables(self) -> Set[str]:
        qvars = set()
        for triple in self.__edges:
            s, _, o = triple.split()
            if s.startswith("?"): qvars.add(s)
            if o.startswith("?"): qvars.add(o)
        
        return qvars
    
    def update(self, triples: Set[str]):
        for triple in triples:
            if triple in self.__edges:
                continue
            
            self.__edges.append(triple)

            s, _, o = str(triple).split()
            for candidate in [s, o]:
                if candidate not in self.__children and candidate.startswith("?"):
                    self.__children.append(candidate)
                
    def __repr__(self) -> str:
        #return f"Subquery: {{parent: {self.__parent}, children: {self.__children}, triples: {self.__edges}}}"
        return f"Subquery: {{triples: {self.__edges}}}"
    
    def __str__(self) -> str:
        # return f"""SELECT * WHERE {{
        #     { ' '.join([e + ' . ' for e in self.__edges]) }
        # }} LIMIT 1
        # """
        return f"Subquery: {{triples: {self.__edges}}}"
    
    def stringify(self, isAskQuery=False, **kwargs): 
        
        type_assertions = [] 
        if kwargs.get("type_assertions") is not None:
            type_assertions = kwargs.get("type_assertions").values()
                    
        query_template = f"""SELECT * WHERE {{
            { ' '.join([e + ' . ' for e in type_assertions ]) }
            { ' '.join([e + ' . ' for e in self.__edges]) }
        }}
        """
        
        if isAskQuery: query_template += " LIMIT 1"
        query = translate_query(query_template, **kwargs)    
        
        return query