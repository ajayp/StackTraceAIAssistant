## StackTraceAIAssistant Proof of Concept (PoC)

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-app-brightgreen)](https://streamlit.io/)

### Table of Contents

- [Introduction](#introduction)
- [Why use AST?](#why-use-ast)
- [Features](#features)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Using the Application](#using-the-application)
- [Project Structure](#project-structure)
- [Future Improvements (PoC Ideas)](#future-improvements-poc-ideas)
- [Example](#example)

### Introduction

This project is a PoC demonstrating how to retrieve relevant code snippets and similar historical stack traces based on a new error event. It utilizes sentence embeddings for semantic similarity and Abstract Syntax Tree (AST) analysis to identify structurally relevant code.

The PoC is implemented as an interactive Streamlit application, making it easy to visualize the retrieval process.

### Why use AST?

While sentence embeddings capture the *semantic meaning* of code (what it does), AST helps understand its *structure* (how it's organized). For error context, structural information is key: errors related to exceptions are often near `try/except`, input errors near validation `if`s, and processing errors near loops. AST analysis complements semantic search by identifying code structurally relevant to the error type.

### How AST works:  

1.  **Parsing:** Source code is parsed into a tree structure.
2.  **Tree Generation:** Nodes represent code constructs (functions, loops, conditionals, etc.).
3.  **Traversal:** The tree is walked to find specific node types.
4.  **Feature Extraction:** Relevant structural features (like the presence of `try/except` or function calls) are collected.

### Features

*   **Semantic Search:** Uses a pre-trained SentenceTransformer model (`all-MiniLM-L6-v2`) to embed code snippets and stack trace frames.
*   **AST Feature Extraction:** Analyzes code snippets to identify structural elements like `try/except` blocks, conditionals (`if`), and loops (`for`, `while`).
*   **Interactive UI:** A Streamlit application allows users to input a simulated error message and stack trace, adjust retrieval parameters (similarity threshold, top-k results), and view the retrieved context.
*   **Contextual Relevance:** Highlights code snippets that are structurally relevant based on simple AST heuristics (e.g., showing `try/except` blocks if the error message suggests an exception).

### Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # If this is part of a larger repo
    # git clone <your-repo-url>
    # cd <your-repo-directory>
    ```

2.  **Install dependencies:**
    This project requires Python 3.7+. It uses `streamlit`, `sentence-transformers`, `torch`, and `numpy`.
    ```bash
    pip install streamlit sentence-transformers torch numpy
    ```
    *Note: Installing `torch` might require specific commands depending on your operating system and CUDA requirements. Refer to the PyTorch installation guide for details.*

### How to Run

1.  Navigate to the directory containing `app.py` in your terminal.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Your web browser should open automatically to the Streamlit application.

### Using the Application

1.  **Input Error:** In the main area, you can either manually type an error message and stack trace or select one of the provided examples from the dropdown.
2.  **Configure Retrieval:** Use the sidebar sliders and number inputs to adjust the `Similarity Threshold` (how similar items must be to be considered relevant) and `Top-K Results per Frame` (how many results to show for each frame in the input stack trace).
3.  **Retrieve Context:** Click the "🔍 Retrieve Context" button.
4.  **View Results:** The application will display "Relevant Code Snippets" and "Similar Past Stack Frames" found in the simulated database based on your input and settings. Expand the sections to view the full code or stack frame content.

### Project Structure

*   `app.py`: Contains all the code for the Streamlit application, including data simulation, AST parsing, embedding, and retrieval logic.

### Future Improvements (PoC Ideas)

*   Integrate with a real vector database (e.g., Chroma, Weaviate, Pinecone).
*   Load code snippets and stack traces from actual project files or logs.
*   Implement more sophisticated AST feature extraction and weighting.
*   Combine semantic similarity and AST features in a more advanced ranking algorithm.
*   Use a larger, domain-specific embedding model.
*   Add functionality to link retrieved code back to specific lines in the original files.
*   Incorporate Large Language Models (LLMs) to summarize the retrieved context or suggest potential fixes.

### Example

<img width="1604" alt="image" src="https://github.com/user-attachments/assets/1f655977-cb7c-4e11-bb68-55301513bea1" />

