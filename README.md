# LLM + RAG: Setup, Training & Demo

This repository provides guidelines and resources for setting up, training, and deploying large language models (LLMs) with a Retrieval-Augmented Generation (RAG) system. The guide covers:

- Installing Ollama to run your LLMs.
- Collecting and preparing training data.
- Fine-tuning an embedding model.
- Deploying a demo that leverages a RAG system.

## Setup & Installation:

### Ollama:
The easiest way to install Ollama is by using Docker. You can find the readily available image on Docker Hub here: [hub.docker.com/r/ollama/ollama](https://hub.docker.com/r/ollama/ollama).

**Note**: If your system/server does not support Docker (e.g., ComputeCanada), you can download the Ollama binary instead: [Ollama Binary Installation](https://github.com/ollama/ollama/blob/main/docs/linux.md#download-the-ollama-binary).

### Embedding Model:
For the embedding model, I used `mxbai-embed-large-v1`, which is available here: [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).

Choosing a good and powerful model as as starting point is very important as the LLM's output will rely heavily on the context found by the embedding model.

## Data Collection:

### Webscraping
There are many guides online so I will keep this short. The code I generally use is BeautifulSoup based and I like to use [curl_cffi](https://github.com/yifeikong/curl_cffi). to handle my requests.

As an example, I included the code I used to collect all the data from Interpares ITrust AI, an interdisciplinary project aiming to leverage Artificial Intelligence to support the availability of public records.

To use:
- 1. Create a config file with 'init_config.py'
- 2. In the config file there are options for filtering and avoiding urls in your website of choice.
- 3. Select main content body with search_contents and adding a dictionary with keys such as "tag", "class", and "style". By default it will find all popular elements such as h1, h2, p, and li.
- 4. Run with python scrape.py --config_file config/your_config.json

### Query generation

We generated queries using the [GPT-4o](https://chatgpt.com/?model=gpt-4o) model. These queries are instrumental for training the embedding model. Below is the prompt we used for generating queries in JSON format:

```markdown
You are a smart and helpful assistant. Your mission is to write one text retrieval example for this task in JSON format. The JSON object must contain the following keys:
"query": a string, a random user search query specified by the retrieval task.
"positive": This will be provided to you. It is a list of strings, each representing a positive example of a document that should be retrieved by the search query.

Please adhere to the following guidelines:
- The "query" should be a random user search query.
- The "query" should be paragraph-based, in at least 10 words, understandable with some effort or ambiguity, and diverse in topic.
- Query should be strongly related to the "positive" examples given to you.
- Your input will be just the text which is to be considered as the positive examples.
```

Original Data: 
```json
{
    "text": "InterPARES Trust is an international research project funded by the Social Sciences and Humanities Research Council of Canada. The project is designed to investigate issues concerning digital records and data entrusted to the Internet. The project is a collaborative effort of researchers from many countries and disciplines. The project is designed to investigate issues concerning digital records and data entrusted to the Internet. The project is a collaborative effort of researchers from many countries and disciplines."
}
```

To generate the queries we use the following code:

```shell
python gen_query.py --input_file data.jsonl --output_file query.jsonl
```

Generated Query:
```json
{
    "query": "What is the InterPARES Trust project about?",
    "positive": [
        "InterPARES Trust is an international research project funded by the Social Sciences and Humanities Research Council of Canada."
    ]
}
```

## Training

### Data Preprocessing: Create a dataset by Mining Hard Negatives

A very useful guide to further preprocess your data and train your model can be found here: https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune.

What we want is to create a dataset by mining hard negatives. This approach significantly improves the model's effectiveness in a target domain.

The code works by using the following command:

```bash
python mine_hn.py \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file query.jsonl \
--output_file query_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```

- `input_file`: json data for finetuning. This script will retrieve top-k documents for each query,
and random sample negatives from the top-k documents (not including the positive documents).
- `output_file`: path to save JSON data with mined hard negatives for finetuning
- `negative_number`: the number of sampled negatives
- `range_for_sampling`: where to sample negative. For example, `2-100` means sampling `negative_number` negatives from top2-top200 documents. **You can set larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top60-300 passages)**
- `candidate_pool`: The pool to retrieval. The default value is None, and this script will retrieve from the combination of all `neg` in `input_file`.
- `use_gpu_for_searching`: whether to use faiss-gpu to retrieve negatives.

Original Data:
```json
{
    "query": "What is the InterPARES Trust project about?",
    "positive": [
        "InterPARES Trust is an international research project funded by the Social Sciences and Humanities Research Council of Canada."
    ]
}
```
Mined Hard Negative Data:
```json
{
    "query": "What is the InterPARES Trust project about?",
    "positive": [
        "InterPARES Trust is an international research project funded by the Social Sciences and Humanities Research Council of Canada."
    ],
    "negative": [
        "InterPARES Trust is a small local initiative with no international involvement or collaboration.",
        "The project is solely focused on historical paper records, with no interest in digital data or internet-related issues.",
        "InterPARES Trust is a for-profit organization rather than a research project.",
        "The project has no connection to any academic or governmental bodies.",
        "InterPARES Trust is primarily concerned with physical archival storage solutions.",
        "The project operates independently without any partnerships or collaborations.",
        "InterPARES Trust does not engage in any research or investigation activities."
    ]
}
```

### Train

To train the model we use the following command:

```
torchrun --nproc_per_node 1 \
run.py \
--output_dir archivesrag \
--model_name_or_path mixedbread-ai/mxbai-embed-large-v1 \
--train_data query_minedHN.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 32 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "" 
```

**some important arguments**:

- `per_device_train_batch_size`: batch size in training. In most of cases, larger batch size will bring stronger performance. You can expand it by enabling `--fp16`, `--deepspeed ./df_config.json` (df_config.json can refer to [ds_config.json](./ds_config.json)), `--gradient_checkpointing`, etc.
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one positive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.
- `learning_rate`: select a appropriate for your model. Recommend 1e-5/2e-5/3e-5 for large/base/small-scale.
- `temperature`: It will influence the distribution of similarity scores. **Recommended value: 0.01-0.1.**
- `query_max_len`: max length for query. Please set it according the average length of queries in your data.
- `passage_max_len`: max length for passage. Please set it according the average length of passages in your data.
- `query_instruction_for_retrieval`: instruction for query, which will be added to each query. You also can set it `""` to add nothing to query.
- `use_inbatch_neg`: use passages in the same batch as negatives. Default value is True.
- `save_steps`: for setting how many training steps to save a checkpoint.

# Usage & Deployment:
Included are `rag.py` and `gradioapp.py`.

- 'rag.py': Integrates your LLM and RAG system.
- 'gradioapp.py': Deploys your demo to an public url. With functionality to ingest pdf documents on the go.

gradioapp.py has the following arguments:
- `model`: llm name. Note: you may need to download or register your model with Ollama to use. Use a command such as `ollama pull llama3`.
- `embedding_model`: Name or path to your embedding model.
- `data_path`: Path to your PDFs.

## Useful Tools to Consider:

### 1. **LangChain**
LangChain is a versatile library designed to make building applications that integrate language models with other components, such as retrieval systems, more accessible. It supports various applications, from chatbots to complex reasoning tasks.
- **GitHub Repository**: [LangChain](https://github.com/langchain-ai/langchain)
- **Documentation**: [LangChain Docs](https://docs.langchain.com/)

### 2. **LLaMA_Index**
LLaMA_Index integrates retrieval systems directly with LLaMA models, providing efficient and scalable solutions for information retrieval combined with generative models.
- **GitHub Repository**: [LLaMA_Index](https://github.com/run-llama/llama_index)
- **Documentation**: [Read the Docs](https://docs.llamaindex.ai/en/stable/)

### 3. **Haystack**
Haystack is an end-to-end framework for building search systems that scale. It allows developers to plug in different components for document storage, retrieval, and question answering systems, perfect for pairing with LLMs.
- **GitHub Repository**: [Haystack](https://github.com/deepset-ai/haystack)
- **Documentation**: [Haystack Docs](https://haystack.deepset.ai/)

### 4. **FAISS**
FAISS (Facebook AI Similarity Search) is an efficient similarity search and clustering of dense vectors. It is particularly useful for creating custom retrieval systems to augment the capabilities of language models.
- **GitHub Repository**: [FAISS](https://github.com/facebookresearch/faiss)
- **Website**: [FAISS Wiki](https://ai.meta.com/tools/faiss/)

### 5. **OpenAI’s Retrieval Augmented Generation**
OpenAI offers a RAG feature that leverages retrieval capabilities alongside their powerful language models. This approach allows developers to enrich the responses of the model by retrieving and referencing relevant documents.
- **Documentation**: [OpenAI API Docs](https://platform.openai.com/docs/overview)

### 6. **Gradio**
Gradio makes it easy to create custom UI components for machine learning models. It allows users to quickly build and share demos that include features like inputs for text, images, and outputs like labels, texts, images, etc. Gradio is particularly user-friendly for rapid prototyping and sharing your LLM models with retrieval capabilities.
- **GitHub Repository**: [Gradio](https://github.com/gradio-app/gradio)
- **Documentation**: [Gradio Docs](https://gradio.app/docs/)

### 7. **Streamlit**
Streamlit is a powerful tool to turn data scripts into shareable web apps in minutes. It allows developers to build highly interactive applications specifically tailored for showcasing machine learning models, including LLMs with RAG. Streamlit's simple syntax and efficient use of Python make it ideal for fast deployment.
- **GitHub Repository**: [Streamlit](https://github.com/streamlit/streamlit)
- **Documentation**: [Streamlit Docs](https://docs.streamlit.io/)