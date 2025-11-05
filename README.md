# NEO4J Demo Project

Small demo that converts text to a knowledge graph and/or loads a movie CSV into Neo4j, then runs Cypher QA using Gemini (Google) via LangChain integrations.

## Files
- [app.py](app.py) — convert text → graph, push to Neo4j and run QA. Key symbols: [`app.graph`](app.py), [`app.llm`](app.py), [`app.chain`](app.py)
- [app1.py](app1.py) — load sample movie CSV into Neo4j, clear DB, run sample QA. Key symbols: [`app1.graph`](app1.py), [`app1.llm`](app1.py), [`app1.chain`](app1.py)
- [.env](.env) — environment variables (API keys & Neo4j credentials). Do NOT commit this file.

## Requirements
- Python 3.8+
- Neo4j AuraDB or local Neo4j instance
- Google Generative AI API key

Recommended packages (examples):
```sh
pip install python-dotenv neo4j langchain langchain-google-genai langchain-community langchain-experimental

Setup
Copy or edit the existing .env with your credentials:

GOOGLE_API_KEY
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
Install Python dependencies (see above).
