# Contract Q&A RAG Project

Welcome to the Contract Q&A RAG Project! This project aims to develop a Retrieval-Augmented Generation (RAG) system for Contract Question & Answer (Q&A) tasks. The system will leverage hybrid LLM technology to provide high-precision legal expertise in contract analysis and interpretation.

## Overview

Lizzy AI is an early-stage Israeli startup focused on developing the next-generation contract AI. The ultimate goal is to create a fully autonomous artificial contract lawyer capable of drafting, reviewing, and negotiating contracts independently. This project marks the initial step towards achieving this goal by building a powerful contract assistant through a RAG system.

## Project Structure

The project follows a structured organization to facilitate development, experimentation, and collaboration. Here's a brief overview of the directory structure:

- **data/**: Contains raw, processed, and labeled contract data, as well as pre-trained word embeddings.
- **models/**: Stores trained retriever and generator models.
- **src/**: Houses source code for data preprocessing, model training, fine-tuning, and evaluation.
- **utils/**: Includes utility functions for text processing and evaluation metrics.
- **config/**: Stores configuration files for model hyperparameters and training settings.
- **notebooks/**: Jupyter notebooks for data exploration, model training, and evaluation.
- **requirements.txt**: Lists Python dependencies required for the project.
- **README.md**: You are here! Detailed documentation about the project.

## Getting Started

To get started with the project, follow these steps:

1. Set up a Python environment and install dependencies listed in `requirements.txt`.
2. Populate the `data/` directory with raw contract documents.
3. Preprocess the data using scripts in `src/` and store processed data in `data/processed_data/`.
4. Train the retriever and generator models using scripts in `src/`.
5. Fine-tune the RAG model for Contract Q&A tasks.
6. Evaluate the performance of the RAG system using evaluation scripts in `src/`.
7. Iterate on the model and data based on evaluation results to improve performance.
8. Refer to Jupyter notebooks in `notebooks/` for detailed examples and experimentation.

## Contributing

Contributions to the project are welcome! If you'd like to contribute, please follow these guidelines:

- Fork the repository and create a new branch for your feature or bug fix.
- Make your changes and submit a pull request detailing the modifications.
- Ensure that your code follows the project's coding style and conventions.
- Write descriptive commit messages and documentation for your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to Lizzy AI for initiating this project and providing guidance throughout its development.
