# ResumeLens

ResumeLens is a local AI-powered resume review system that compares a PDF resume against a target job description and provides evidence-based suggestions for improving the application.

## Features

- Extracts resume content from PDF files
- Analyses resumes against job descriptions
- Identifies matching and missing skills
- Reviews technical and soft-skill evidence
- Suggests resume improvements
- Runs locally using Qwen3.5-4B
- Uses 4-bit quantization for efficient GPU inference

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Qwen3.5-4B
- BitsAndBytes
- PyMuPDF

## Current Pipeline

```text
Resume PDF
    ↓
PyMuPDF
    ↓
Extracted Text
    +
Job Description
    ↓
Qwen3.5-4B
    ↓
Resume Review