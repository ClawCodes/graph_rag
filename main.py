import os

import pandas as pd
from llama_index.core import Document

from graphrag.extractors import GraphRAGExtractor, KG_TRIPLET_EXTRACT_TMPL, parse_fn
from graphrag.query import GraphRAGQueryEngine
from graphrag.utils import load_llm, QWEN_SMALL
from graphrag.store import GraphRAGStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from dotenv import load_dotenv

load_dotenv()

def main():
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )[:50]

    documents = [
        Document(text=f"{row['title']}: {row['text']}")
        for i, row in news.iterrows()
    ]

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    llm = load_llm(QWEN_SMALL)

    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=2,
        parse_fn=parse_fn,
    )

    # Note: used to be `Neo4jPGStore`
    graph_store = GraphRAGStore(
        model=llm, username="neo4j", password=os.getenv("PASSWORD"), url="bolt://localhost:7687"
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
        embed_model=embed_model,
    )

    index.property_graph_store.build_communities()

    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=10,
    )

    #TODO: fix query, calling OpenAI
    response = query_engine.query(
        "What are the main news discussed in the document?"
    )

    print(response)


if __name__ == '__main__':
    main()