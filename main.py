import os
from argparse import ArgumentParser
from typing import Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from graphrag.extractors import KG_TRIPLET_EXTRACT_TMPL, GraphRAGExtractor, parse_fn
from graphrag.query import GraphRAGQueryEngine
from graphrag.store import GraphRAGStore
from graphrag.utils import QWEN_SMALL, load_llm

load_dotenv()


def load_models(model) -> Tuple[LLM, HuggingFaceEmbedding]:
    llm = load_llm(model)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    return llm, embed_model


def load_index(
    llm: LLM,
    embed_model: HuggingFaceEmbedding,
    nodes: Optional[list[BaseNode]] = None,
    extractor: Optional[GraphRAGExtractor] = None,
) -> PropertyGraphIndex:
    pw = os.getenv("PASSWORD")
    if pw is None:
        raise EnvironmentError("Please set the environment variable 'PASSWORD'")

    graph_store = GraphRAGStore(
        model=llm,
        username="neo4j",
        password=pw,
        url="bolt://localhost:7687",
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
            property_graph_store=graph_store, embed_model=embed_model, llm=llm
        )

    return index


def create_query_engine(llm: LLM, index: PropertyGraphIndex) -> GraphRAGQueryEngine:
    return GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=10,
    )


def _query(engine):
    response = engine.query("What are the main news discussed in the document?")

    print(response)


def query_from_raw() -> None:
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )[:50]

    documents = [
        Document(text=f"{row['title']}: {row['text']}") for i, row in news.iterrows()
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

    _query(query_engine)


def query_existing() -> None:
    llm, embed_model = load_models(QWEN_SMALL)

    index = load_index(llm, embed_model)

    query_engine = create_query_engine(llm, index)

    _query(query_engine)


def main(from_raw: bool) -> None:
    if from_raw:
        query_from_raw()
    else:
        query_existing()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--from-raw", action="store_true")
    args = parser.parse_args()

    main(args.from_raw)
