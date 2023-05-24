import rdflib
import click
from SPARQLWrapper import SPARQLWrapper, JSON
import sys

class Node:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

class Edge:
    def __init__(self, subject: Node, predicate: str, object: Node):
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def __str__(self) -> str:
        return self.subject.value + " " + self.predicate + " " + self.object.value

class QueryTree:
    def __init__(self, query_txt: str):

        self.query = query_txt

        triples = set()

        nodes = set()
        edges = set()

        for line in query_txt.splitlines(keepends=True):
            if line.strip().startswith("#") or \
                line.strip().startswith("PREFIX") or \
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

            triples.add(line.strip())

        for triple in triples:
            sep_triple = str(triple).split()

            ref_subject = None
            ref_object = None

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
    
    def __repr__(self) -> str:
        return self.query
    
    def get_edges(self, vrtx: str) -> set:
        edges = set()
        for edge in self.edges:
            if str(edge.subject) == vrtx or str(edge.object) == vrtx:
                edges.add(edge)
        return edges
    
    def get_triple_patterns(self) -> set:
        return self.edges
    
    def add_edge(self, edge: Edge):
        self.query.replace(".\n}", str(edge))
        self.edges.add(edge)
        self.nodes.add(edge.subject)
        self.nodes.add(edge.object)
        return self
    
    def get_relevant_source(self, endpoint: SPARQLWrapper) -> JSON:
        query = self.query
        query.replace("WHERE {", "WHERE { GRAPH ?g {")
        query.replace(".\n}", ". }\n }")
        endpoint.setQuery(query)
        return endpoint.query().convert()

def create_subquery(edge: Edge) -> QueryTree:
    return QueryTree("SELECT * \nWHERE {\n " + str(edge.subject) + " " + edge.predicate + " " + str(edge.object) + " .\n}")

def get_parent_subquery(vrtx: str, subqueries: set) -> QueryTree:
    for subquery in subqueries:
        for edge in subquery.get_edges(vrtx):
            if str(edge.object) == vrtx:
                return subquery
            
def can_be_added_to_subquery(subquery: QueryTree, edge: Edge, V: set, endpoint: SPARQLWrapper) -> bool:
    results_1 = subquery.get_relevant_source(endpoint)
    results_2 = create_subquery(edge).get_relevant_source(endpoint)
    return results_1 == results_2

def add_to_subquery(subquery: QueryTree, edge: Edge):
    return subquery.add_edge(edge)

def merge_subquery(subqueries: set) -> set:
    return subqueries

def estimate_cost(subqueries: set) -> int:
    return len(subqueries)

@click.group
def cli():
    pass

@cli.command()
@click.argument("query", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument("global-join-var", type=str, nargs=-1)
@click.argument("out-result", type=click.Path(exists=False, file_okay=True, dir_okay=True))
@click.argument("endpoint", type=str)
@click.pass_context
def decompose(ctx: click.Context, query, global_join_var, out_result, endpoint):
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    subqueries = set()
    query_txt = ""

    with open(query, "r") as qfs:
        query_txt = qfs.read()

    Q = QueryTree(query_txt)

    best_decomposition = set()
    min_decomp_cost = sys.maxsize

    if not query_txt:
        return Q
       
    triples = Q.get_triple_patterns()

    for j_var_i in global_join_var:
        visited_triples = set()
        nodes = set()
        subqueries = set()
        nodes.add(j_var_i)

        while nodes:
            vrtx = nodes.pop()
            edges = Q.get_edges(vrtx)

            if not subqueries:

                for edge_i in edges:

                    if edge_i in visited_triples:
                        continue

                    sq = create_subquery(edge_i)
                    subqueries.add(sq)
                    nodes.add(edge_i.object)
                    visited_triples.add(edge_i)
                continue

            parent_sq = get_parent_subquery(vrtx, subqueries)
            for edge_i in edges:

                if edge_i in visited_triples:
                    continue

                if can_be_added_to_subquery(parent_sq, edge_i, global_join_var, sparql):
                    parent_sq = add_to_subquery(parent_sq, edge_i)

                else:
                    sq = create_subquery(edge_i)
                    subqueries.add(sq)
                nodes.add(edge_i.object)
                visited_triples.add(edge_i)

        if visited_triples == Q:
            subqueries = merge_subquery(subqueries)
            cost = estimate_cost(subqueries)

            if cost < min_decomp_cost:
                best_decomposition = subqueries
                min_decomp_cost = cost

    with open(out_result, 'w') as out:
        for subquery in best_decomposition:
            out.write(subquery + "\n")

if __name__ == "__main__":
    cli()