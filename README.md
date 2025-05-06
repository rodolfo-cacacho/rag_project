# 🇩🇪 RAG for Architecture and Sustainability in Germany  
**Retrieval-Augmented Generation for Green Building Regulations and Funding Programs**

This repository presents the solution I developed as part of my Master's Thesis at **THWS Master AI Würzburg**, focused on improving access to and understanding of complex regulatory documents related to sustainable architecture and energy-efficient building programs in Germany.

## 🧠 Problem Statement: Navigating Complex Building Regulations

Public and private stakeholders involved in sustainable construction often face lengthy, technical, and fragmented documentation. Regulatory texts, funding guidelines, and model specifications are dispersed across different formats and sources. This complicates efficient access to relevant information for architects, planners, and citizens.

### Key Challenges

- Parsing diverse document formats (guidelines, FAQs, technical models)
- Handling OCR and unstructured or tabular content in PDFs
- Managing **German-language** regulatory content
- Merging and chunking documents while preserving semantic structure
- Retrieving relevant passages via semantic + keyword hybrid search
- Ensuring traceability from responses back to source context
- Supporting explainability and document transparency

## 💡 My Solution

I developed a modular **Retrieval-Augmented Generation (RAG)** system that enables scalable and context-aware retrieval of information from regulatory texts, with both **semantic** and **keyword-based** filtering.

### 🧾 Document Processing Pipeline

- Input documents (guidelines, FAQs, specifications) were processed as **PDFs**
- **OCR and structure analysis** for complex layouts, including tables as images
- Documents were **merged, cleaned, and semantically chunked**
- Custom preprocessing for **BM25 indexing** and **dense embeddings**

### 🔎 Hybrid Search & Semantic Retrieval

- **Keyword search (BM25)** for high recall and handling rare terms
- **Dense vector embeddings** for semantic understanding
- A configurable **hybrid scoring system** balances both methods
- Enhanced filtering using document type and thematic keywords

### 🧠 Generation with Context-Aware Prompts

- Retrieved passages are passed to an LLM to answer user queries
- Context includes document type, source metadata, and regulatory scope
- Answers are **linked back to source documents** for transparency

## ✅ Impact

My system facilitates **faster access** to relevant regulatory content, supporting architects, engineers, and consultants in planning and compliance tasks. It improves:
- **Efficiency** in navigating funding requirements
- **Transparency** through traceable context
- **Scalability** for future regulatory updates

## 🛠️ Technologies Used

- **SentenceTransformers** for dense embeddings (German model)
- **BM25** with custom vocabulary extraction for keyword search
- **OCR & PDF parsing** (Adobe API, PyMuPDF)
- **Langchain & FAISS/Pinecone** for RAG pipeline
- **LLMs** for question answering and summarization

## 🗂️ Document Types Handled

- BEG Guidelines (WG, NWG, EM)
- FAQs and Explanatory Notes
- Technical System Models and Specifications

## 📫 Contact

For questions or collaboration inquiries, feel free to reach out.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
