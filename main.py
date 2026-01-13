import os
from typing import Optional

import pandas as pd
from llama_index.core import Document

from graphrag.extractors import GraphRAGExtractor, KG_TRIPLET_EXTRACT_TMPL, parse_fn
from graphrag.query import GraphRAGQueryEngine
from graphrag.utils import load_llm, QWEN_SMALL
from graphrag.store import GraphRAGStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PropertyGraphIndex
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import LLM

from dotenv import load_dotenv

load_dotenv()


def load_models(model):
    llm = load_llm(model)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    return llm, embed_model


def load_index(llm: LLM, embed_model: HuggingFaceEmbedding, nodes: Optional[list[BaseNode]] = None,
               extractor: Optional[GraphRAGExtractor] = None) -> PropertyGraphIndex:
    graph_store = GraphRAGStore(
        model=llm, username="neo4j", password=os.getenv("PASSWORD"), url="bolt://localhost:7687"
    )

    if nodes is not None and extractor is not None:
        index = PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[extractor],
            property_graph_store=graph_store,
            show_progress=True,
            embed_model=embed_model,
        )
    else:
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_model=embed_model,
            llm=llm
        )

    return index

def create_query_engine(llm: LLM, index: PropertyGraphIndex) -> GraphRAGQueryEngine:
    return GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=10,
    )

def query_from_raw():
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

    llm, embed_model = load_models(QWEN_SMALL)

    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=2,
        parse_fn=parse_fn,
    )

    index = load_index(llm, embed_model, nodes, extractor=kg_extractor)

    index.property_graph_store.build_communities()

    query_engine = create_query_engine(llm, index)

    response = query_engine.query(
        "What are the main news discussed in the document?"
    )

    print(response)

def query_existing():
    llm, embed_model = load_models(QWEN_SMALL)

    index = load_index(llm, embed_model)

    query_engine = create_query_engine(llm, index)

    response = query_engine.query(
        "What are the main news discussed in the document?"
    )

    print(response)

def main():
    query_existing()

if __name__ == '__main__':
    main()
