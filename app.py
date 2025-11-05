
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain
import os

# ----------------------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------------------
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# ----------------------------------------------------------------
# 2️⃣ Connect to Neo4j Database
# ----------------------------------------------------------------
graph = Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password,
)

# ----------------------------------------------------------------
# 3️⃣ Initialize Gemini LLM
# ----------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or gemini-1.5-flash (faster, cheaper)
    google_api_key=google_api_key
)

# ----------------------------------------------------------------
# 4️⃣ Input text to convert into a Knowledge Graph
# ----------------------------------------------------------------
text = """
Elon Reeve Musk (born June 28, 1971) is a businessman and investor known for his key roles in space
company SpaceX and automotive company Tesla, Inc. Other involvements include ownership of X Corp.,
formerly Twitter, and his role in the founding of The Boring Company, xAI, Neuralink and OpenAI.
He is one of the wealthiest people in the world; as of July 2024, Forbes estimates his net worth to be
US$221 billion. Musk was born in Pretoria to Maye and engineer Errol Musk, and briefly attended
the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through
his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada.
Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics
and physics. He moved to California in 1995 to attend Stanford University, but dropped out after
two days and, with his brother Kimbal, co-founded online city guide software company Zip2.
"""

documents = [Document(page_content=text)]

# ----------------------------------------------------------------
# 5️⃣ Convert Text → Graph Structure using Gemini
# ----------------------------------------------------------------
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# ----------------------------------------------------------------
# 6️⃣ Upload Graph Data to Neo4j
# ----------------------------------------------------------------
graph.add_graph_documents(graph_documents)
print("✅ Graph data successfully pushed to Neo4j!")

print("\nExtracted Relationships:")
for rel in graph_documents[0].relationships:
    print(f"{rel.source.id} -[{rel.type}]-> {rel.target.id}")

# ----------------------------------------------------------------
# 7️⃣ Ask questions about the data (QA over Neo4j)
# ----------------------------------------------------------------
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

# Example question
query = "When was Elon Reeve Musk?"
response = chain.invoke({"query": query})

print("\nQuestion:", query)
print("Answer:", response)

