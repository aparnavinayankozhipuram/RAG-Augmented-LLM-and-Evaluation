# RAG-Augmented LLM

This repository contains resources and evaluation notebooks for a Retrieval-Augmented Generation (RAG) pipeline using Large Language Models (LLMs), focusing on experimentation and benchmarking. Example notebooks utilize models such as Mistral-7B and Llama-3-1-8B for information retrieval, augmentation, and generation tasks.

## Project Overview

RAG-Augmented LLM combines advanced language models with external knowledge retrieval to produce more informed and context-rich responses.  
Key features include:
- Integration of retrieval and augmentation steps with LLMs
- Example workflows in Jupyter notebooks
- Evaluation scripts for benchmarking model performance

# RAG-Augmented LLM: Human Nutrition Example

This repository demonstrates an end-to-end Retrieval-Augmented Generation (RAG) pipeline using Large Language Models (LLMs), such as Mistral-7B, for document retrieval and question answering in the domain of human nutrition.

## Project Overview

- **RAG Pipeline:** Combines document retrieval (using FAISS, HuggingFace embeddings) with LLM-based augmentation (Mistral-7B, LlamaCpp).
- **Use Case:** Answers complex questions about human nutrition by retrieving relevant information from an open-access textbook.
- **Modular:** Includes document loading, chunking, embedding, vector store creation, and QA chain setup.

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- GPU (recommended, for faster LLM inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aparnavinayankozhipuram/Mistral-7B-Evaluation.git
   cd Mistral-7B-Evaluation
   ```

2. **Install dependencies:**
   You can use the following to install all required libraries. (Or use the included `requirements.txt` if provided.)
   ```bash
   pip install langchain torch sentence_transformers faiss-cpu huggingface-hub pypdf accelerate llama-cpp-python bitsandbytes chromadb langchain_community ragas arxiv pymupdf wandb tiktoken
   ```

### Usage

1. **Download the example PDF**  
   The script will automatically download [Human Nutrition (OER Hawaii)](https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf) if not present.

2. **Run the pipeline:**  
   Edit and execute `rag_augmented_llm.py` (provided in this repo) to:
   - Load and split the PDF
   - Generate embeddings and vector store
   - Set up the LLM (Mistral-7B or another HuggingFace-compatible model)
   - Run a sample QA query

   Example:
   ```bash
   python rag_augmented_llm.py
   ```

## Project Structure

- `rag_augmented_llm.py` — Main script for RAG workflow (standalone, refactored from Colab notebook)
- `README.md` — Project documentation
- `requirements.txt` — (Optional) List of required dependencies

## Customization

- **Change the input PDF:** Replace the URL or path in the script.
- **Change the LLM model:** Update `llm_model_path` to use another HuggingFace or Llama-compatible model.
- **Ask different questions:** Modify the `query` string in the script.

## References

- [LangChain](https://github.com/langchain-ai/langchain)
- [Mistral-7B](https://github.com/mistralai/Mistral-7B)
- [Human Nutrition PDF](https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)

## License

This project is MIT licensed. See [LICENSE](LICENSE) for details.

## Contact

For questions or suggestions, please [open an issue](https://github.com/aparnavinayankozhipuram/Mistral-7B-Evaluation/issues).
## Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions, bug fixes, or improvements.


## Acknowledgements

- [Mistral-7B](https://github.com/mistralai/Mistral-7B)
- [Llama-3](https://github.com/meta-llama/llama3)
- Other open-source contributors

## Contact

For questions or collaboration, reach out via [GitHub Issues](https://github.com/aparnavinayankozhipuram/Mistral-7B-Evaluation/issues).
