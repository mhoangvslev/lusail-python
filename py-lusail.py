from copy import deepcopy
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Set, Tuple
import click
import networkx as nx
import numpy as np
import pandas as pd
from pyparsing import ParseResults
from lusail.QueryTree import QueryTree
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateAlgebra, translateQuery
import urllib.request
from lusail.Subquery import Subquery

from lusail.utils import exec_query_on_relevant_sources, get_pair_triples, get_triple_from_edge

proxy_support = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

@click.group
def cli():
    pass

@cli.command()
@click.argument("queryfile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def parse_query(queryfile) -> List[str]:
    with open(queryfile) as qfs:
        query_tree = QueryTree(qfs.read())
        return query_tree
    
@cli.command()
@click.argument("query", type=click.STRING)
@click.pass_context
def get_global_join_variables(ctx: click.Context, query):
    parse_tree = QueryTree(query)
        
    def formulate_subj_obj_query(var, subjTriple, objTriple):
        query_template = f"""
        SELECT {var} WHERE {{
            {parse_tree._type_assertions[var] + " . " if parse_tree._type_assertions.get(var) is not None else ''}
            {subjTriple} .
            FILTER NOT EXISTS {{ 
                SELECT {var} WHERE {{
                    {objTriple} .
                }} 
            }} . 
        }} LIMIT 1
        """
        
        ptree: ParseResults = parseQuery(query_template)        
        ptree[0].extend(parse_tree._prefixes)
        ptree[1]["where"]["part"].extend(parse_tree._filters)
                            
        query_algebra = translateQuery(ptree)
        query = translateAlgebra(query_algebra)
        
        return query

    def formulate_pairwise_query(var, pair):
        first, second = pair
        
        query_template = f"""SELECT {var} WHERE {{
            {{
                SELECT {var} WHERE {{
                    {parse_tree._type_assertions[var] + " . " if parse_tree._type_assertions.get(var) is not None else ''}
                    {first} .
                    FILTER NOT EXISTS {{ 
                        SELECT * WHERE {{
                            {second} . 
                        }} 
                    }} 
                }}
            }} UNION {{
                SELECT {var} WHERE {{
                    {parse_tree._type_assertions[var] + " . " if parse_tree._type_assertions.get(var) is not None else ''}
                    {second} .
                    FILTER NOT EXISTS {{ 
                        SELECT * WHERE {{
                            {first} . 
                        }} 
                    }} 
                }}
            }} 
        }}  LIMIT 1
        """
        
        ptree: ParseResults = parseQuery(query_template)        
        ptree[0].extend(parse_tree._prefixes)
        ptree[1]["where"]["part"].extend(parse_tree._filters)
                                    
        query_algebra = translateQuery(ptree)
        query = translateAlgebra(query_algebra)
        
        return query
    
    vars = [ node for node, degree in parse_tree._hyperGraph.degree() if str(node).startswith("?") and degree > 1 ]
    chkQueries = []
    global_join_variables = {}
    
    for var in vars:
        triples = parse_tree.get_edges(var)
        triples = [ get_triple_from_edge(e) for e in triples ]
        pairWiseTriples = get_pair_triples(triples)
        joinVars = False

        for pair in pairWiseTriples:
            first, second = pair
            
            first_ss = parse_tree.get_relevant_sources(first)
            second_ss = parse_tree.get_relevant_sources(second)
                            
            if set(first_ss) != set(second_ss):
                if global_join_variables.get(var) is None: global_join_variables[var] = []
                global_join_variables[var].append(pair)
                joinVars = True
        
        if joinVars: continue

        # Record the position of var amongst the triples. 0 = subject, 1 = object
        var_pos = set()
        subjTriples = []
        objTriples = []
        for triple in triples:
            triple_as_list = triple.split(" ")
            idx = triple_as_list.index(var)
            if idx == 0: subjTriples.append(triple)
            elif idx == 2: objTriples.append(triple)
            else: raise RuntimeError("Join variable should never be a predicate")
            var_pos.add(idx)

        # if var is subject only or object only
        if len(var_pos) == 1:
            #print("Subject only/object only")
            for pair in pairWiseTriples:
                chkQuery = formulate_pairwise_query(var, pair)
                chkQueries.append((chkQuery, var, pair[0], pair[1]))

        # if var is subject and object
        if len(var_pos) == 2:
            # print("Object/subject")
            #print(subjTriples, objTriples, list(product(subjTriples, objTriples)))
            for subjTriple, objTriple in product(subjTriples, objTriples):
                # print(var, objTriple, subjTriple)
                # Reverse order in the pair so it would be object/subject join
                chkQuery = formulate_subj_obj_query(var, objTriple, subjTriple)
                chkQueries.append((chkQuery, var, objTriple, subjTriple))
        
    for chkQuery, var, first, second in chkQueries:
        first_relevant_sources = set(parse_tree.get_relevant_sources(first))
        second_relevant_sources = set(parse_tree.get_relevant_sources(second))
        relSources = first_relevant_sources.intersection(second_relevant_sources)
        
        diff = exec_query_on_relevant_sources(chkQuery, relSources)
        
        if not diff.empty:
            if global_join_variables.get(var) is None: global_join_variables[var] = []
            global_join_variables[var].append((first, second))
            
    # print(list(global_join_variables.keys()))
    return list(global_join_variables.keys())

# @cli.command()
# @click.argument("queryfile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
# @click.pass_context
# def get_join_variables(ctx: click.Context, queryfile) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
#     parse_tree: QueryTree = ctx.invoke(parse_query, queryfile=queryfile)
#     vars = [ node for node, degree in parse_tree.hyperGraph.degree() if degree > 1 ]
#     global_join_variables = {}
    
#     for var in vars:
#         triples = parse_tree.get_edges(var)
#         triples = [ get_triple_from_edge(e) for e in triples ]

#         # Record the position of var amongst the triples. 0 = subject, 1 = object
#         for triple in triples:
#             triple_as_list = triple.split(" ")
#             idx = triple_as_list.index(var)
#             if idx == 2 and triple_as_list[1] == "owl:sameAs":
#                 if global_join_variables.get(var) is None: global_join_variables[var] = []
#                 global_join_variables[var].append((triple))
            
#     return global_join_variables

@cli.command()
@click.argument("queryfile", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.pass_context
def decompose_query(ctx: click.Context, queryfile):
    
    query = None
    with open(queryfile, "r") as qfs:
        query = qfs.read()
    
    parse_tree = QueryTree(query)
    
    def create_subquery(edge: str, parent: str):
        return Subquery(edge, parent)
    
    def get_parent_subquery(vrtx, subqueries: List[Subquery]) -> Subquery:
        for subquery in subqueries:
            if vrtx in subquery.get_children():
                return subquery
                    
        raise RuntimeError(f"Could not find {vrtx}!")
                
    def can_be_added_to_subquery(subquery: Subquery, edge: str, global_join_variables: List[str]) -> bool:
        results_1 = parse_tree.get_relevant_sources(subquery)
        results_2 = parse_tree.get_relevant_sources(edge)
        if set(results_1) != set(results_2):
            return False
        
        new_subQ = deepcopy(subquery)
        new_subQ.update([edge])

        new_subQ_str = new_subQ.stringify(prefixes=parse_tree._prefixes, filters=parse_tree._filters)
        gjv_new_subQ = ctx.invoke(get_global_join_variables, query=new_subQ_str)

        # Check if the newly merged subquery doesn't add new gjv        
        if len(set(gjv_new_subQ)) > 0 and set(gjv_new_subQ).issubset(set(global_join_variables)):
            return False

        print("GJV:", global_join_variables, "and", gjv_new_subQ)
        return True

    def merge_subquery(subqueries: List[Subquery]) -> Set[Subquery]:
        merged_subqueries: Set[Subquery] = set(subqueries)
        
        for subQ1, subQ2 in combinations(subqueries, 2):
            
            # If both subqueries share the same relSources, ...
            rel_sources_sq1 = parse_tree.get_relevant_sources(subQ1)
            rel_sources_sq2 = parse_tree.get_relevant_sources(subQ2)
            if set(rel_sources_sq1) != set(rel_sources_sq2):
                continue

            # ... and they share join variables
            common_join_vars = subQ1.get_join_variables().intersection(subQ2.get_join_variables()) 
            # print("Common jv:", common_join_vars)           
            if len(common_join_vars) == 0:
                continue
           
            # ... and the shared join variables are not global join vars
            newSubQ = Subquery.merge(subQ1, subQ2)
            newSubQ_str = newSubQ.stringify(prefixes=parse_tree._prefixes, filters=parse_tree._filters)
            newSubQ_gjv = set(ctx.invoke(get_global_join_variables, query=newSubQ_str))
            if len(newSubQ_gjv) > 0:
                continue
            
            merged_subqueries.add(newSubQ)
            if subQ1 in merged_subqueries: merged_subqueries.remove(subQ1)
            if subQ2 in merged_subqueries: merged_subqueries.remove(subQ2)
            
        return merged_subqueries

    def estimate_cost(subqueries: Set[Subquery]) -> float:
        """Apply the Chauvenet’s criterion for detecting and rejecting outliers
        before computing \mu and \sigma.
        
        Any subquery sqi with cardinality C(sqi) > mu_C + sigma_C is delayed

        Args:
            subqueries (Set[Subquery]): _description_
            relevant_sources (List[str]): _description_

        Returns:
            float: _description_
        """
        
        def get_cardinality(triple, endpoint):
            cardinality_query = f"""SELECT (COUNT(*) as ?card) WHERE {{
                { triple } .
            }}
            """
            
            ptree = parseQuery(cardinality_query)
            ptree[0].extend(parse_tree._prefixes)
            ptree[1]["where"]["part"].extend(parse_tree._filters)
                    
            query_algebra = translateQuery(ptree)
            query = translateAlgebra(query_algebra)
            
            results = exec_query_on_relevant_sources(query, [endpoint])
             
            return results.values.item()
        
        cost = []
        
        for sq in subqueries:
            cost_sq = -np.inf
            qvars = sq.get_variables()
            endpoints = parse_tree.get_relevant_sources(sq)
            for qvar in qvars:
                cost_sq_v = 0
                for tp_i, tp_j in combinations(sq.get_triple_patterns(), 2):
                    s_i, _, o_i = tp_i.split()
                    s_j, _, o_j = tp_j.split()
                    
                    join_qvars = set([s_i, o_i]).intersection(set([s_j, o_j]))
                    if len(join_qvars) == 1 and join_qvars.pop() == qvar:
                        for ep in endpoints:
                            cost_tpi_ep = get_cardinality(tp_i, ep)
                            cost_tpj_ep = get_cardinality(tp_j, ep)
                            cost_sq_v_ep = min(cost_tpi_ep, cost_tpj_ep)
                            cost_sq_v += cost_sq_v_ep
                if cost_sq_v > cost_sq:
                    cost_sq = cost_sq_v
            cost.append(cost_sq)
        return cost
    
    def get_dest_node(edge, node):
        s, _, o = str(edge).split()
        if node == s:
            return o
        elif node == o:
            return s
        else:
            raise ValueError(f"Something is wrong")
    
    # BEGIN
    global_join_variables = ctx.invoke(get_global_join_variables, query=query)
    join_variables = [ node for node, degree in parse_tree._hyperGraph.degree() if str(node).startswith("?") and degree > 1 ]
            
    if len(global_join_variables) == 0:
        return parse_tree._query
    
    best_decomposition: List[Subquery] = None
    min_decomp_cost = np.inf
    triples = parse_tree.get_triple_patterns()

    for jvar in global_join_variables:
        visited_triples = set()
        nodes = []
        subqueries = []
        nodes.append(jvar)
        
        itr = 0
        
        while len(nodes) > 0:
            print(f"--- ITERATION {itr} ---")
            vrtx = nodes.pop()
            print("Vertex:", vrtx)
            edges = parse_tree.get_edges(vrtx)
            edges = [ get_triple_from_edge(e) for e in edges ]
            print("Edges:", edges)
                                                
            if len(subqueries) == 0:
                for edge in edges:
                    if edge in visited_triples: continue
                    
                    dest_node = get_dest_node(edge, vrtx)
                    if dest_node.startswith("?") and dest_node in join_variables:
                        nodes.append(dest_node)
                        
                    sq = create_subquery(edge, vrtx)
                    subqueries.append(sq)
                    print(f"Create subquery {sq}")
                    visited_triples.add(edge) 
                print("Children nodes:", nodes)   
                itr += 1
                continue
                        
            parent_sq = get_parent_subquery(vrtx, subqueries)
                        
            for edge in edges:
                if edge in visited_triples: continue
                if can_be_added_to_subquery(parent_sq, edge, global_join_variables):
                    parent_sq.update([edge])
                    print(f"Add {edge} to subquery {parent_sq}")
                else:
                    sq = create_subquery(edge, vrtx)
                    subqueries.append(sq)
                    print(f"Create subquery {sq}")
                    
                dest_node = get_dest_node(edge, vrtx)
                if dest_node.startswith("?") and dest_node in join_variables:
                    nodes.append(dest_node)
                    
                visited_triples.add(edge)
            
            print("Children nodes:", nodes)
            itr += 1
                    
        print(sorted(visited_triples), len(visited_triples))
        print(sorted(triples), len(triples))                        
        if set(visited_triples) == set(triples):
            decomposition = merge_subquery(subqueries)
                        
            cost = sum(estimate_cost(subqueries))
            print("Subqueries:", subqueries)
            print("Decomposition:", decomposition, "cost:", cost)
        
            if cost < min_decomp_cost:
                best_decomposition = decomposition
                min_decomp_cost = cost        

    print(best_decomposition)
    return (best_decomposition, min_decomp_cost)
if __name__ == "__main__":
    cli()