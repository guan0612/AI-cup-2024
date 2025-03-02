# 🏆 AI Cup 2024 E.SUN Financial Question Answering Challenge Rank : 62/487

## 📝 Project Introduction

This project represents our solution for the "AI Cup 2024 E.SUN Financial Question Answering Challenge." We've try to optimize the retrieve performance on financial domain question answering.

## 🔧 Technical Architecture

- **Text Processing**: CKIP BERT, Monpa, Jieba
- **Retrieval System**: Enhanced BM25
- **OCR Engine**: Tesseract
- **Evaluation Framework**: Precision metrics

## 📊 Performance Comparison

| Dataset | Baseline Performance | Our Best Performance | Improvement |
|---------|---------------------|----------------------|------------|
| FAQ | 0.70 | 0.90 | +28.6% |
| Finance | 0.70 | 0.90 | +28.6% |
| Insurance | 0.60 | 0.70 | +16.7% |

## 🔬 Methodology

### 1. Word Segmentation Exploration

We experimented with multiple Chinese word segmentation approaches:
- **Jieba** (baseline): Standard Chinese text segmentation
- **Monpa**: Specialized for Traditional Chinese
- **BERT-based tokenization**: Contextual tokenization
- **CKIP BERT**: Domain-adaptive tokenization

While these approaches showed incremental improvements, they weren't sufficient to achieve our performance targets.

### 2. Document Processing Enhancement

Upon analysis of the dataset, we discovered that many PDF documents contained embedded images with critical financial information. We implemented:

- OCR processing pipeline for image text extraction
- Text normalization for financial terminology
- Document structure analysis to maintain contextual relationships

Despite these efforts, the improvements remained modest, likely due to OCR quality limitations.

### 3. POS-Based Feature Selection

Our breakthrough came from implementing CKIP's Part-of-Speech tagging system:

By selectively extracting terms with specific POS tags relevant to financial domain (nouns, proper nouns, financial terms), we:
1. Reduced noise in the document representation
2. Enhanced the prominence of domain-specific terminology
3. Improved the precision of BM25 retrieval

This approach yielded our best results, with significant performance improvements across all datasets.
