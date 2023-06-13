from copy import deepcopy
from itertools import combinations, product
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Set, Tuple
import click
import networkx as nx
import numpy as np
import pandas as pd
from pyparsing import ParseResults
from lusail.QueryTree import QueryTree

import urllib.request
from lusail.Subquery import Subquery

from lusail.utils import exec_query_on_relevant_sources, get_pair_triples, get_triple_from_edge, translate_query

proxy_support = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

@click.group
def cli():
    pass

@cli.command()
@click.argument("queryfile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def parse_query(queryfile):
    with open(queryfile) as qfs:
        q = qfs.read()
        query_tree = QueryTree(q)
        return query_tree
    
@cli.command()
@click.argument("query", type=click.STRING)
@click.pass_context
def get_global_join_variables(ctx: click.Context, query) -> Set[str]:
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
                
        query = translate_query(query_template, prefixes=parse_tree._prefixes, filters=parse_tree._filters)
        
        return query

    def formulate_pairwise_query(var, pair):
        first, second = pair
        
        first_type_assertions = [ parse_tree._type_assertions.get(item) + " . " for item in first.split() if item in parse_tree._type_assertions ]
        second_type_assertions = [ parse_tree._type_assertions.get(item) + " . " for item in second.split() if item in parse_tree._type_assertions ]
        
        query_template = f"""SELECT {var} WHERE {{
            {{
                SELECT {var} WHERE {{
                    {''.join(first_type_assertions)}
                    {first} .
                    FILTER NOT EXISTS {{ 
                        SELECT * WHERE {{
                            {''.join(second_type_assertions)}
                            {second} . 
                        }} 
                    }} 
                }}
            }} UNION {{
                SELECT {var} WHERE {{
                    {''.join(second_type_assertions)}
                    {second} .
                    FILTER NOT EXISTS {{ 
                        SELECT * WHERE {{
                            {''.join(first_type_assertions)}
                            {first} . 
                        }} 
                    }} 
                }}
            }} 
        }}  LIMIT 1
        """
                
        query = translate_query(query_template, prefixes=parse_tree._prefixes, filters=parse_tree._filters)
        
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
                    
            if first_ss != second_ss:
                print(f"Variable: {var}, First: '{first}', Second: '{second}', not sharing the same relevant sources")
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
            # print("Subject only/object only")
            for pair in pairWiseTriples:
                chkQuery = formulate_pairwise_query(var, pair)
                chkQueries.append((chkQuery, var, pair[0], pair[1]))

        # if var is subject and object
        if len(var_pos) == 2:
            # print("Object/subject")
            # print(subjTriples, objTriples, list(product(subjTriples, objTriples)))
            for subjTriple, objTriple in product(subjTriples, objTriples):
                # print(var, objTriple, subjTriple)
                # Reverse order in the pair so it would be object/subject join
                chkQuery = formulate_subj_obj_query(var, objTriple, subjTriple)
                chkQueries.append((chkQuery, var, objTriple, subjTriple))
        
    for chkQuery, var, first, second in chkQueries:
        first_relevant_sources = parse_tree.get_relevant_sources(first)
        second_relevant_sources = parse_tree.get_relevant_sources(second)
        relSources = first_relevant_sources.intersection(second_relevant_sources)
        
        diff = exec_query_on_relevant_sources(chkQuery, relSources)        
        if not diff.empty:
            print(var, first, second, diff, chkQuery)
            if global_join_variables.get(var) is None: global_join_variables[var] = []
            global_join_variables[var].append((first, second))
            
    gjvs = set(global_join_variables.keys())
    #parse_tree.draw(gjv=gjvs)
    print(f"Global join variables: {gjvs}")
    return gjvs

# @cli.command()
# @click.argument("query", type=click.STRING)
# @click.pass_context
# def get_global_join_variables(ctx: click.Context, query):
#     parse_tree = QueryTree(query)
#     vars = [ node for node, degree in parse_tree._hyperGraph.degree() if degree > 1 ]
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
    
#     gjvs = list(global_join_variables.keys())
#     # parse_tree.draw(gjv=gjvs)
#     print(f"Global join variables: {gjvs}")
#     return gjvs

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
    
    def get_parent_subquery(vrtx, subqueries: Set[Subquery]) -> Subquery:
        for subquery in subqueries:
            if vrtx in subquery.get_children():
                return subquery
                    
        raise RuntimeError(f"Could not find {vrtx}!")
                
    def can_be_added_to_subquery(subquery: Subquery, edge: str, global_join_variables: Set[str]) -> bool:
        results_1 = parse_tree.get_relevant_sources(subquery)
        results_2 = parse_tree.get_relevant_sources(edge)
        if results_1 != results_2:
            return False
        
        new_subQ = deepcopy(subquery)
        new_subQ.update([edge])

        new_subQ_str = new_subQ.stringify(prefixes=parse_tree._prefixes, filters=parse_tree._filters, type_assertions=parse_tree._type_assertions)
        gjv_new_subQ: Set[str] = ctx.invoke(get_global_join_variables, query=new_subQ_str)

        # Check if the newly merged subquery doesn't add new gjv        
        if len(gjv_new_subQ) > 0 and gjv_new_subQ.issubset(global_join_variables):
            return False

        print("GJV:", global_join_variables, "and", gjv_new_subQ)
        return True

    def merge_subquery(subqueries: Set[Subquery]) -> Set[Subquery]:
        """ loops through the set of subqueries and merges a pair of 
            subqueries if they have common variables, the same relevant sources, 
            and no pair of triple patterns from both subqueries has a common variable that is global.
        """
        merged_subqueries = deepcopy(subqueries)
        
        for subQ1, subQ2 in combinations(merged_subqueries, 2):
            
            # If both subqueries share the same relSources, ...
            rel_sources_sq1 = parse_tree.get_relevant_sources(subQ1)
            rel_sources_sq2 = parse_tree.get_relevant_sources(subQ2)
            if rel_sources_sq1 != rel_sources_sq2:
                print("Merge cond1 unsatisfied: subqueries do not share relevant sources")
                continue

            # ... and they share common variables
            common_vars = subQ1.get_variables().intersection(subQ2.get_variables())          
            if len(common_vars) == 0:
                print("Merge cond 2 unsatisfied: no common variables:", common_vars, subQ1.get_variables(), subQ2.get_variables())  
                continue
           
            # ... and no pair of triple patterns from both subqueries has a common variable that is global 
            some_pair_has_global_variables = False
            for pair in product(subQ1.get_triple_patterns(), subQ2.get_triple_patterns()):
                query_template = f"""SELECT * WHERE {{
                    { ' '.join([e + ' . ' for e in parse_tree._type_assertions.values()]) }
                    { ' '.join([e + ' . ' for e in pair]) }
                }}
                """
                                
                query = translate_query(query_template, prefixes=parse_tree._prefixes, filters=parse_tree._filters)
                new_gjvs = ctx.invoke(get_global_join_variables, query=query)
                if len(new_gjvs) > 0:
                    some_pair_has_global_variables = True
                    print(f"Merge cond 3 unsatisfied: {pair} has {new_gjvs} that are global!")
                    break
            
            if some_pair_has_global_variables:
                continue
                
            newSubQ: Subquery = Subquery.merge(subQ1, subQ2)            
            merged_subqueries.add(newSubQ)
            if subQ1 in merged_subqueries: merged_subqueries.remove(subQ1)
            if subQ2 in merged_subqueries: merged_subqueries.remove(subQ2)
            
        return merged_subqueries

    def estimate_cost(subqueries: Set[Subquery]) -> float:
        
        def get_cardinality(triple, endpoint):
            s, _, o = triple.split()
            query_template = f"""SELECT (COUNT(*) as ?card) WHERE {{
                #{ ' '.join([ parse_tree._type_assertions[qvar] + ' . ' for qvar in [s, o] if qvar in parse_tree._type_assertions.keys() ]) }
                { triple } .
            }}
            """
            
            query = translate_query(query_template, prefixes=parse_tree._prefixes, filters=parse_tree._filters)
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
    
    best_decomposition: Set[Subquery] = None
    min_decomp_cost = np.inf
    triples = parse_tree.get_triple_patterns()

    for jvar in global_join_variables:
        visited_triples = set()
        nodes = []
        subqueries = set()
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
                    subqueries.add(sq)
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
                    print(f"Add '{edge}' to subquery {parent_sq}")
                else:
                    sq = create_subquery(edge, vrtx)
                    subqueries.add(sq)
                    print(f"Create subquery {sq}")
                    
                dest_node = get_dest_node(edge, vrtx)
                if dest_node.startswith("?") and dest_node in join_variables:
                    nodes.append(dest_node)
                    
                visited_triples.add(edge)
            
            print("Children nodes:", nodes)
            itr += 1
       
        print(visited_triples, triples)                                  
        if visited_triples == triples:
            print("Subqueries:")
            pprint(subqueries)
            
            decomposition = merge_subquery(subqueries)       
            cost = sum(estimate_cost(decomposition))
            print(f"Decomposition of cost {cost}:")
            pprint(decomposition)
        
            if cost < min_decomp_cost:
                best_decomposition = decomposition
                min_decomp_cost = cost        

    # print(parse_tree.bind_subqueries(best_decomposition))
    return (best_decomposition, min_decomp_cost)
if __name__ == "__main__":
    cli()