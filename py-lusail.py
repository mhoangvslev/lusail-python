import rdflib
import click
from SPARQLWrapper import SPARQLWrapper, JSON
import sys
import re
import json

class Node:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value
    
    def is_var(self) -> bool:
        return self.value.startswith('?')

class Edge:
    def __init__(self, subject: Node, predicate: str, object: Node):
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def __repr__(self) -> str:
        return self.subject.value + " " + self.predicate + " " + self.object.value

    def __str__(self) -> str:
        return self.subject.value + " " + self.predicate + " " + self.object.value

class QueryTree:
    def __init__(self, query_txt: str):

        self.query = query_txt

        triples = set()

        nodes = set()
        edges = set()

        prefix = set()

        for line in query_txt.splitlines(keepends=True):
            if line.strip().startswith("#") or \
                line.strip().startswith("SELECT") or \
                line.strip().startswith("WHERE") or \
                line.strip().startswith("{") or \
                line.strip().startswith("}") or \
                line.strip().startswith("OPTIONAL") or \
                line.strip().startswith("FILTER") or \
                line.strip().startswith("ORDER") or \
                line.strip().startswith("LIMIT") or \
                line.strip() == '':
                continue

            elif line.strip().startswith("PREFIX"):
                prefix.add(line.strip())

            else:
                triples.add(line.strip())

        for triple in triples:
            sep_triple = str(triple).split()

            ref_subject = Node("dummy")
            ref_object = Node("dummy")

            if sep_triple[0] not in nodes:
                ref_subject = Node(sep_triple[0])
                nodes.add(ref_subject)
            else:
                for node in nodes:
                    if sep_triple[0] != node:
                        continue
                    ref_subject = node

            if sep_triple[2] not in nodes:
                ref_object = Node(sep_triple[2])
                nodes.add(ref_object)
            else:
                for node in nodes:
                    if sep_triple[2] != node:
                        continue
                    ref_object = node

            edges.add(Edge(ref_subject, sep_triple[1], ref_object))

        self.nodes = nodes
        self.edges = edges
        self.prefix = prefix
    
    def __repr__(self) -> str:
        return self.query

    def __str__(self) -> str:
        return self.query
    
    def get_edges(self, vrtx: Node) -> set:
        edges = set()
        for edge in self.edges:
            #print(f"vrtx: {vrtx} -> subject: {edge.subject} -> object: {edge.object}")
            if str(edge.subject) == str(vrtx) or str(edge.object) == str(vrtx):
                edges.add(edge)
        #print(f"[{len(self.edges)}] - vrtx: {vrtx} -> edges: {edges}")
        return edges
    
    def get_triple_patterns(self) -> set:
        return self.edges
    
    def add_edge(self, edge: Edge):
        self.query = re.sub("\.\n}", ".\n" + str(edge) + " .\n}", self.query)
        #print(self.query)
        self.edges.add(edge)
        self.nodes.add(edge.subject)
        self.nodes.add(edge.object)
        return self
    
    def get_relevant_source(self, endpoint: SPARQLWrapper) -> JSON:
        query = self.query
        query = re.sub("SELECT \*", "SELECT DISTINCT ?g", query)
        query = re.sub("WHERE {", "WHERE { GRAPH ?g {", query)
        query = re.sub(".\n}", ". }\n }", query)
        endpoint.setQuery(query)
        return endpoint.query().convert()
    
    def get_var(self) -> set:
        nodes = set()
        for node in self.nodes:
            if node.is_var():
                nodes.add(node)
        return nodes

def create_subquery(edge: Edge, prefixs: set) -> QueryTree:
    str_prefix = ""
    for prefix in prefixs:
        str_prefix+= str(prefix) + "\n"
    return QueryTree(str_prefix + "SELECT * \nWHERE {\n " + str(edge.subject) + " " + edge.predicate + " " + str(edge.object) + " .\n}")

def get_parent_subquery(vrtx: Node, subqueries: set) -> QueryTree:
    for subquery in subqueries:
        for edge in subquery.get_edges(vrtx):
            if str(edge.subject) == str(vrtx) or str(edge.object) == str(vrtx):
                return subquery
            
def can_be_added_to_subquery(subquery: QueryTree, edge: Edge, V: set, endpoint: SPARQLWrapper) -> bool:
    #print(subquery)
    results_1 = subquery.get_relevant_source(endpoint)
    results_2 = create_subquery(edge, subquery.prefix).get_relevant_source(endpoint)
    return results_1 == results_2

def add_to_subquery(subquery: QueryTree, edge: Edge) -> QueryTree:
    subquery.add_edge(edge)
    return subquery

def merge_subquery(subqueries: set, V: set, endpoint: SPARQLWrapper) -> set:
    copy_subqueries = subqueries
    merged_subqueries = set()
    ref_merged_edges = set()
    for subquery in subqueries:
        for copy_subquery in copy_subqueries:
            common_var = subquery.get_var().intersection(copy_subquery.get_var())
            if common_var:
                if subquery.get_relevant_source(endpoint) == copy_subquery.get_relevant_source(endpoint):
                    if not common_var.intersection(V):
                        merged_edges = subquery.edges.union(copy_subquery.edges)
                        if merged_edges.intersection(ref_merged_edges):
                            ref_merged_edges = ref_merged_edges.union(merged_edges)
                            str_prefix = ""
                            for prefix in subquery.prefix:
                                str_prefix+= str(prefix) + "\n"
                            merged_subquery = str_prefix + "SELECT * \nWHERE {\n "
                            for edge in merged_edges:
                                merged_subquery += str(edge) + " .\n"
                            merged_subquery += "}"
                            merged_subqueries.add(merged_subquery)
                        else:
                            merged_subqueries.add(subquery)
                            merged_subqueries.add(copy_subquery)
                    else:
                        merged_subqueries.add(subquery)
                        merged_subqueries.add(copy_subquery)
                else:
                    merged_subqueries.add(subquery)
                    merged_subqueries.add(copy_subquery)
            else:
                merged_subqueries.add(subquery)
                merged_subqueries.add(copy_subquery)

    return merged_subqueries

def estimate_cost(subqueries: set) -> int:
    return len(subqueries)

@click.group
def cli():
    pass

@cli.command()
@click.argument("query", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--local-join-var", type=str, multiple=True)
@click.option("--global-join-var", type=str, multiple=True)
@click.argument("out-result", type=click.Path(exists=False, file_okay=True, dir_okay=True))
@click.argument("endpoint", type=str)
@click.pass_context
def decompose(ctx: click.Context, query, local_join_var, global_join_var, out_result, endpoint):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    LJV = set(Node(local_join_var_i) for local_join_var_i in local_join_var)
    GJV = set(Node(global_join_var_i) for global_join_var_i in global_join_var)

    JV = LJV.union(GJV)

    subqueries = set()
    query_txt = ""

    with open(query, "r") as qfs:
        query_txt = qfs.read()

    Q = QueryTree(query_txt)

    best_decomposition = set()
    min_decomp_cost = sys.maxsize

    if not GJV:
        best_decomposition.add(Q)
       
    triples = Q.get_triple_patterns()

    for j_var_i in GJV:
        visited_triples = set()
        nodes = set()
        subqueries = set()
        nodes.add(j_var_i)

        while nodes:
            vrtx = nodes.pop()
            #print(f"SELECTED NODE: {vrtx}")
            edges = Q.get_edges(vrtx)
            #print(f"LINKED EDGES: {edges}")

            if not subqueries:

                for edge_i in edges:

                    if edge_i in visited_triples:
                        continue

                    sq = create_subquery(edge_i, Q.prefix)

                    subqueries.add(sq)
                    if str(edge_i.object) == str(vrtx):
                        nodes.add(edge_i.subject)
                    else:
                        nodes.add(edge_i.object)
                    #print(f"ADD NODE: {edge_i.object}")
                    visited_triples.add(edge_i)
                    #print(f"VISITED EDGE: {edge_i}")
                continue

            parent_sq = get_parent_subquery(vrtx, subqueries)

            #print(f"PARENT SUBQUERY: \n\n{parent_sq}\n")

            for edge_i in edges:

                if edge_i in visited_triples:
                    continue

                if can_be_added_to_subquery(parent_sq, edge_i, GJV, sparql):
                    parent_sq = add_to_subquery(parent_sq, edge_i)
                    #print(parent_sq)

                else:
                    sq = create_subquery(edge_i, Q.prefix)
                    subqueries.add(sq)

                if str(edge_i.object) == str(vrtx):
                    nodes.add(edge_i.subject)
                else:
                    nodes.add(edge_i.object)
                #print(f"ADD NODE: {edge_i.object}")
                visited_triples.add(edge_i)
                #print(f"VISITED EDGE: {edge_i}")

        #print(visited_triples == triples)
        print(f"visited_triples: {len(visited_triples)} == triples: {len(triples)}")

        if visited_triples == triples:
            subqueries = merge_subquery(subqueries, GJV, sparql)
            #print(subqueries)
            cost = estimate_cost(subqueries)

            if cost < min_decomp_cost and cost > 0:
                best_decomposition = subqueries
                min_decomp_cost = cost

    with open(out_result, 'w') as out:
        for subquery in best_decomposition:
            out.write(str(subquery) + "\n")

if __name__ == "__main__":
    cli()