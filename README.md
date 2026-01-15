### Install

This project uses [poetry](https://python-poetry.org/) for package management.
Install packages with:

```bash
poetry install
```

You will also need to install [ollama](https://ollama.com/download) as it uses a local LLM.
Install `qwen2.5:3b`

```bash
ollama pull qwen2.5:3b
```

### Run

If you have not run this before, the graph DB will not be cached in a local `data/` directory.
To pull neo4j docker image and build container, download the data, and set up the graph store, run:

```bash
invoke kb && poetry run python main.py -r
```

If you already ran the above command, you can use the local store and remove the `-r` flag:

```bash
poetry run python main.py
```

### Query

You can pass your query for the corpus using the `-q` flag

```bash
poetry run python main.py -q "Please tell me about common themes in the news"
```

Or just use the default query that is automatically passed by providing no arguments.
The default query is: "What are the main news discussed in the document?".
