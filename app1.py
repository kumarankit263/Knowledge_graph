from langchain_community.graphs import Neo4jGraph
from dotenv  import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.chains import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
import os

load_dotenv()

google_api_key=os.getenv("GOOGLE_API_KEY")
neo4j_uri=os.getenv("NEO4J_URI")
neo4j_username=os.getenv("NEO4J_USERNAME")
neo4j_password=os.getenv("NEO4J_PASSWORD")

# connect to Neo4j Database

graph=Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password,
)

print("✅ Connected to Neo4j Database!")

graph.query("MATCH (n) DETACH DELETE n")
print("🧹 Cleared all old data from Neo4j!")

movie_query = """
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv' AS row

MERGE (m:Movie {id: row.movieId})
SET 
  m.released = date(row.released),
  m.title = row.title,
  m.imdbRating = toFloat(row.imdbRating)

FOREACH (director IN split(row.director, '|') |
    MERGE (p:Person {name: trim(director)})
    MERGE (p)-[:DIRECTED]->(m))

FOREACH (actor IN split(row.actors, '|') |
    MERGE (p:Person {name: trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))

FOREACH (genre IN split(row.genres, '|') |
    MERGE (g:Genre {name: trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""
graph.query(movie_query)
print("✅ Movie data successfully loaded into Neo4j!")

print("\n🔍 Total nodes:")
print(graph.query("MATCH (n) RETURN COUNT(n) AS node_count"))

print("\n🎬 Sample Movies:")
print(graph.query("MATCH (m:Movie) RETURN m.title, m.imdbRating LIMIT 5"))

print("\n🎭 Sample Relationships:")
print(graph.query("MATCH (a:Person)-[r]->(m:Movie) RETURN a.name, type(r), m.title LIMIT 5"))
# initialize Gemini LLM

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key
)

print("🤖 Gemini model initialized!")
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    allow_dangerous_requests=True,  # required for GraphCypherQAChain
    verbose=True
)
print("🔗 GraphCypherQAChain ready!")

response = chain.invoke({"query": "Who directed the movie GoldenEye?"})
print("\n🎯 Response from Gemini:")
print(response)

# Try another:
response = chain.invoke({"query": "List top 5 movies with highest IMDb ratings."})
print("\n🏆 Top Rated Movies:")
print(response)


