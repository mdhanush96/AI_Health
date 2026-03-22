# MedAI: An Intelligent Healthcare Chatbot Leveraging ClinicalBERT and Retrieval-Augmented Generation for Symptom-Based Disease Prediction and Medical Report Summarization

---

## Authors

**[Author Name]**  
Department of Computer Science and Engineering  
[University / Institution Name]  
[City, State, Country]  
Email: [author@institution.edu]

---

## Abstract

The increasing prevalence of chronic and infectious diseases worldwide demands accessible, intelligent, and reliable healthcare advisory systems. Traditional symptom checkers rely on rule-based decision trees or shallow keyword matching, which lack contextual understanding and clinical reasoning capabilities. This paper presents **MedAI**, a full-stack AI-powered healthcare chatbot that integrates a fine-tuned ClinicalBERT model for symptom-based disease classification with a Retrieval-Augmented Generation (RAG) pipeline for comprehensive medical knowledge delivery. The system processes natural language symptom descriptions, classifies them into one of 22 disease categories with calibrated confidence scores, and augments predictions with semantically retrieved medical knowledge using FAISS vector search and FLAN-T5 generative summarization. The architecture further incorporates a two-pass symptom verification mechanism that re-ranks ClinicalBERT predictions by matching reported symptoms against a curated knowledge base using Jaccard similarity and synonym-aware phrase matching. The system supports multi-modal medical report summarization through multi-pass T5 question answering, regex-based entity extraction, and SciSpacy biomedical NER across PDF, image, and text formats. Emergency detection for critical conditions is implemented via keyword scanning with real-time helpline routing. The platform is deployed as a responsive web application with a React 18 frontend and Django REST Framework backend, supporting token-based authentication, multi-conversation persistence, and real-time file upload processing. Experimental evaluation on the Disease and Symptoms Dataset from Kaggle demonstrates that the proposed hybrid ClinicalBERT–RAG approach achieves 80.6% classification accuracy across 22 disease categories, outperforming traditional ML baselines including Naive Bayes, SVM, Random Forest, and Logistic Regression. The symptom verification layer provides improved prediction relevance through weighted score fusion. The system provides clinically informative responses while maintaining appropriate medical disclaimers and emergency escalation protocols.

**Keywords:** Healthcare Chatbot, ClinicalBERT, Retrieval-Augmented Generation, Natural Language Processing, Disease Prediction, Medical NER, FAISS Vector Search, FLAN-T5, Symptom Classification, Report Summarization

---

## 1. Introduction

### 1.1 Problem Statement

The global healthcare system faces mounting pressure from increasing patient volumes, a shortage of medical professionals, and delayed access to preliminary medical guidance. According to the World Health Organization (WHO), there is a global deficit of approximately 18 million health workers, most acutely felt in low- and middle-income countries. Patients frequently experience prolonged wait times for initial consultations, during which symptoms may deteriorate. The need for accessible, reliable, and intelligent preliminary medical advisory tools has become a critical area of research in healthcare informatics.

### 1.2 Importance of AI Healthcare Chatbots

Artificial Intelligence (AI)-driven healthcare chatbots offer a scalable solution for preliminary symptom assessment and health information delivery. These systems leverage Natural Language Processing (NLP) and Machine Learning (ML) to interpret user-described symptoms and provide disease predictions, medical knowledge, and actionable health guidance. Unlike traditional rule-based symptom checkers that rely on rigid decision trees, modern NLP-based chatbots can understand contextual symptom descriptions, handle linguistic variations, and provide nuanced medical information grounded in clinical knowledge bases.

### 1.3 Limitations of Existing Systems

Current healthcare chatbot implementations exhibit several key limitations:

1. **Rule-Based Approaches:** Systems such as Ada Health and Buoy Health employ decision-tree algorithms requiring exhaustive manual encoding of symptom–disease mappings, resulting in limited scalability and inability to handle novel symptom descriptions.

2. **Shallow ML Models:** Traditional classifiers (Naive Bayes, SVM, Random Forest) applied to symptom classification lack the ability to capture semantic relationships between symptoms, resulting in degraded performance for ambiguous or multi-symptom presentations.

3. **Lack of Contextual Knowledge Delivery:** Most existing systems provide a disease label without comprehensive medical context, leaving users without understanding of causes, treatment options, complications, or when to seek emergency care.

4. **No Report Processing Capability:** Existing chatbots typically cannot process uploaded medical reports (PDF, images) for automated summarization and entity extraction.

5. **Absence of Post-Classification Verification:** Predictions from classification models are presented without validation against curated clinical knowledge, leading to potential misalignment between predicted diseases and reported symptoms.

### 1.4 Motivation

The convergence of transformer-based language models (BERT, GPT), dense retrieval systems (FAISS), and instruction-tuned generative models (FLAN-T5) presents an opportunity to develop healthcare chatbots that combine the classification precision of fine-tuned clinical models with the knowledge depth of retrieval-augmented generation. This research bridges the gap between accurate disease prediction and comprehensive, contextually grounded medical information delivery.

### 1.5 Research Contributions

The principal contributions of this work are:

1. A **hybrid disease prediction framework** combining fine-tuned ClinicalBERT classification with a Retrieval-Augmented Generation (RAG) pipeline using FAISS vector search and FLAN-T5 generation for 22 disease categories.

2. A **two-pass symptom verification mechanism** that re-ranks ClinicalBERT predictions using Jaccard similarity, synonym-aware phrase matching, and weighted score fusion (60% model confidence + 40% symptom match).

3. A **multi-modal medical report summarization engine** supporting PDF, image (OCR), and text input formats with multi-pass T5 question answering, regex-based entity extraction, and SciSpacy biomedical NER.

4. An **emergency detection system** with keyword-based critical condition identification and real-time helpline routing.

5. A **production-ready full-stack deployment** architecture with React 18 frontend, Django REST Framework backend, token-based authentication, multi-conversation management, and GPU-accelerated inference.

---

## 2. Literature Review

### 2.1 Related Work

| # | Paper / System | Method | Limitation |
|---|----------------|--------|------------|
| 1 | Babylon Health (Middleton et al., 2016) | Bayesian Network + probabilistic reasoning | Limited to structured symptom inputs; no free-text support |
| 2 | Ada Health (Razzaki et al., 2018) | Decision tree + rule-based inference | Requires exhaustive manual rule engineering |
| 3 | Buoy Health (Semigran et al., 2015) | Bayesian classifier + symptom checklist | Does not provide detailed medical knowledge |
| 4 | HealthBot (Bates et al., 2019) | Naive Bayes + TF-IDF | Cannot capture semantic relationships |
| 5 | Mandy (Ni et al., 2017) | SVM + keyword matching | Poor generalization to unseen symptoms |
| 6 | MedBot (Xu et al., 2019) | Random Forest + symptom encoding | No post-classification verification; no RAG |
| 7 | BioBERT (Lee et al., 2020) | Pre-trained biomedical BERT for NER | Not fine-tuned for symptom classification |
| 8 | ClinicalBERT (Huang et al., 2019) | BERT pre-trained on clinical notes | General clinical NLP; not applied to chatbots |
| 9 | SymptomNet (Chaudhari et al., 2021) | CNN + LSTM for symptom sequences | Lacks knowledge retrieval; labels only |
| 10 | RAG (Lewis et al., 2020) | Retrieval-Augmented Generation framework | General-purpose; not medical-domain specific |
| 11 | Med-PaLM (Singhal et al., 2023) | Large Language Model for medical QA | Requires massive compute; not open-source |
| 12 | ChatDoctor (Li et al., 2023) | LLaMA fine-tuned on medical dialogues | Hallucination risk; no verification mechanism |

### 2.2 Research Gap

The literature reveals a significant gap between classification-only systems that provide disease predictions without contextual knowledge and large language model (LLM) systems that generate fluent but potentially hallucinated medical content. MedAI addresses this gap by:

- Combining fine-tuned ClinicalBERT classification with grounded knowledge retrieval, ensuring predictions are both accurate and contextually supported.
- Implementing a two-pass symptom verification layer that validates model predictions against curated clinical knowledge.
- Using FAISS vector search with FLAN-T5 generation to produce factually grounded medical explanations rather than relying on unconstrained text generation.

---

## 3. System Architecture

### 3.1 Architectural Overview

MedAI employs a four-layer architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                          │
│              React 18 · React Router v7 · Axios                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │ ChatPage │  │LoginPage │  │Register  │  │  ProfilePage     │  │
│   │          │  │          │  │  Page    │  │  (Health Info)    │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ Components: Sidebar · ChatBox · ChatMessage · ResultCard    │  │
│   │             History · AuthContext                            │  │
│   └──────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                          API LAYER                                 │
│           Django REST Framework · Token Authentication             │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  POST /api/predict-rag/    →  Hybrid prediction + RAG       │  │
│   │  POST /api/predict/        →  ClinicalBERT only             │  │
│   │  GET  /api/health/         →  System diagnostics            │  │
│   │  GET  /api/history/        →  Prediction audit log          │  │
│   │  POST /api/auth/register/  →  User registration             │  │
│   │  POST /api/auth/login/     →  Token authentication          │  │
│   │  POST /api/auth/logout/    →  Token invalidation            │  │
│   │  GET  /api/auth/profile/   →  User profile management       │  │
│   └──────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                      ML INFERENCE LAYER                            │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐    │
│   │ ClinicalBERT │  │  RAG Pipeline │  │ Report Summarizer   │    │
│   │ (Disease     │  │  FAISS +      │  │ T5 QA + SciSpacy   │    │
│   │ Classifier)  │  │  FLAN-T5      │  │ NER + Regex         │    │
│   └──────────────┘  └───────────────┘  └─────────────────────┘    │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐    │
│   │  Symptom     │  │  Medical NER  │  │  Emergency          │    │
│   │  Verifier    │  │  (SciSpacy)   │  │  Detector           │    │
│   └──────────────┘  └───────────────┘  └─────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                  │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐    │
│   │   MySQL DB   │  │ FAISS Vector  │  │  Knowledge Base     │    │
│   │  (medai_db)  │  │   Index       │  │  (22 Disease JSONs) │    │
│   └──────────────┘  └───────────────┘  └─────────────────────┘    │
│   ┌──────────────┐  ┌───────────────┐                              │
│   │ ClinicalBERT │  │  disease_info │                              │
│   │   Weights    │  │    .json      │                              │
│   └──────────────┘  └───────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Technology Stack

| Layer | Technology | Version / Details |
|-------|-----------|-------------------|
| Frontend | React | 18.2.0 |
| Routing | React Router | 7.13.1 |
| HTTP Client | Axios | 1.13.6 (120s timeout) |
| State Management | React Context API | + localStorage persistence |
| Backend | Django REST Framework | Python 3.11.9 |
| Database | MySQL | medai_db |
| Authentication | DRF TokenAuthentication | + SessionAuthentication |
| Classifier | ClinicalBERT | BertForSequenceClassification (~110M params) |
| Embeddings | all-MiniLM-L6-v2 | 384-dimensional |
| Vector Store | FAISS | L2 distance index |
| Generator | FLAN-T5-Base | ~220M params |
| NER | SciSpacy | en_core_sci_md |
| OCR | EasyOCR / Pytesseract | Fallback chain |
| PDF Processing | pdfplumber / PyPDF2 | Fallback chain |

---

## 4. Dataset Description

### 4.1 Primary Dataset

| Property | Value |
|----------|-------|
| **Name** | Disease and Symptoms Dataset |
| **Source** | Kaggle (Choong Qian Zheng) |
| **Format** | JSONL (newline-delimited JSON) |
| **Split** | 80% training / 20% test (stratified) |
| **Classes** | 22 disease categories |
| **Random Seed** | 42 |

**Data Format:**
```json
{
  "input_text": "I have been experiencing continuous high fever, severe headache, and body pain for the past 3 days...",
  "output_text": "dengue"
}
```

### 4.2 Disease Categories

The 22 disease classes span tropical diseases, chronic conditions, infections, and autoimmune disorders:

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Allergy | 12 | Hypertension |
| 2 | Arthritis | 13 | Impetigo |
| 3 | Bronchial Asthma | 14 | Jaundice |
| 4 | Cervical Spondylosis | 15 | Malaria |
| 5 | Chicken Pox | 16 | Migraine |
| 6 | Common Cold | 17 | Peptic Ulcer Disease |
| 7 | Dengue | 18 | Pneumonia |
| 8 | Diabetes | 19 | Psoriasis |
| 9 | Drug Reaction | 20 | Typhoid |
| 10 | Fungal Infection | 21 | Urinary Tract Infection |
| 11 | GERD | 22 | Varicose Veins |

### 4.3 Knowledge Base

A curated knowledge base of 22 structured JSON files provides grounded medical knowledge for each disease, with sections covering:

- **Overview** — Clinical definition, epidemiology, pathophysiology
- **Symptoms** — Detailed presentation, stages, warning signs
- **Causes** — Etiology, risk factors, transmission mechanisms
- **Treatment** — Management strategies, medications, supportive care
- **When to See a Doctor** — Urgency indicators, red-flag symptoms
- **Preventions** — Risk mitigation, lifestyle modifications

**Example Schema** (dengue.json):
```json
{
  "Overview": "Dengue is a mosquito-borne viral infection endemic in >100 countries...",
  "Symptoms": "Clinical spectrum ranges from asymptomatic to severe hemorrhagic...",
  "Causes": "DENV-1/2/3/4 serotypes transmitted by Aedes aegypti...",
  "Treatment": "Supportive care with fluid management and monitoring...",
  "When to See a Doctor": "Prompt medical attention for persistent high fever...",
  "Preventions": "Vector control, mosquito repellents, elimination of breeding sites..."
}
```

---

## 5. Preprocessing and Feature Engineering

### 5.1 Text Preprocessing Pipeline

User symptom input undergoes the following preprocessing:

1. **Input Sanitization:** Strip whitespace, enforce length bounds (3–5000 characters), type validation
2. **Greeting Detection:** 15+ regex patterns identify conversational inputs (e.g., "hello", "how are you") with medical signal keyword checks to distinguish greetings from symptom descriptions
3. **Tokenization:** ClinicalBERT WordPiece tokenizer (vocab size = 28,996) with max sequence length of 128 tokens, padding, and truncation

### 5.2 Medical Report Processing

Multi-modal file ingestion supports three formats with graceful fallback chains:

| Format | Primary Tool | Fallback | Extraction Method |
|--------|-------------|----------|-------------------|
| PDF | pdfplumber | PyPDF2 | Text + table extraction per page |
| Image (PNG/JPG/JPEG) | EasyOCR | Pytesseract | OCR with GPU/CPU support |
| Text (TXT) | UTF-8 decode | — | Direct text extraction |

### 5.3 Report Entity Extraction

Medical reports undergo structured entity extraction via three parallel methods:

1. **SciSpacy Biomedical NER** (`en_core_sci_md`): Entities classified into 6 categories — diseases, medications, symptoms, tests, procedures, anatomy. Drug suffix matching with 28 known pharmaceutical patterns. Noise filtering with 60+ generic term blacklist.

2. **Regex-Based Structured Extraction:** 30+ patterns for lab values with clinical interpretation (e.g., hemoglobin, glucose, WBC, creatinine, cholesterol, HbA1c, TSH), patient demographics, medication schedules, and diagnostic findings.

3. **T5 Multi-Pass Question Answering:** 5 focused clinical questions:
   - "What are the key diagnoses or conditions?"
   - "What are the important lab values and their interpretation?"
   - "What medications are prescribed?"
   - "What are the key recommendations?"
   - "What is the patient's demographic information?"

### 5.4 Feature Engineering for Baseline Models

Traditional ML baselines use TF-IDF vectorization:

| Parameter | Value |
|-----------|-------|
| Max Features | 10,000 |
| N-gram Range | (1, 2) — unigrams + bigrams |
| Sublinear TF | Enabled (log-scaled term frequency) |

---

## 6. Model Architecture

### 6.1 ClinicalBERT Disease Classifier

The primary classifier is a fine-tuned **ClinicalBERT** model (`BertForSequenceClassification`) pre-trained on clinical notes and fine-tuned on the Disease and Symptoms Dataset.

**Architecture Specifications:**

| Parameter | Value |
|-----------|-------|
| Model Type | BERT (BertForSequenceClassification) |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Hidden Layers | 12 |
| Intermediate Size | 3,072 |
| Vocabulary Size | 28,996 |
| Max Position Embeddings | 512 |
| Total Parameters | ~110M |
| Classification Head | Linear (768 → 22) + Softmax |
| Problem Type | single_label_classification |
| Precision | float32 |
| Input Max Length | 128 tokens |
| Activation | GELU |
| Attention Dropout | 0.1 |
| Hidden Dropout | 0.1 |

**Inference Pipeline:**

```
Input Text
    │
    ▼
┌─────────────────────────────┐
│  WordPiece Tokenizer        │
│  (vocab=28,996, max_len=128)│
│  padding + truncation       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  BERT Encoder               │
│  12 Transformer Layers      │
│  768-dim hidden states      │
│  12 attention heads         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  [CLS] Token Pooling        │
│  768-dim representation     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Dense Layer (768 → 22)     │
│  + Softmax Activation       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Top-K Predictions          │
│  (disease, confidence, risk)│
└─────────────────────────────┘
```

**Risk Classification Thresholds:**

| Confidence Range | Risk Level | Color Code |
|------------------|-----------|------------|
| ≥ 70% | High Probability | 🔴 Red (#ef4444) |
| 40%–70% | Moderate Probability | 🟡 Amber (#f59e0b) |
| < 40% | Low Confidence | 🔵 Blue (#3b82f6) |

### 6.2 Thread-Safe Model Loading

Model loading employs a **double-check locking singleton pattern** for thread safety:

```python
_model = None
_tokenizer = None
_lock = threading.Lock()

def load_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:                    # First check (no lock)
        with _lock:                       # Acquire lock
            if _model is None:            # Second check (with lock)
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                _model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
                _model.to(device)
                _model.eval()
    return _model, _tokenizer
```

---

## 7. Retrieval-Augmented Generation (RAG) Pipeline

### 7.1 RAG Architecture Overview

The RAG pipeline augments ClinicalBERT predictions with contextually grounded medical knowledge through three stages: embedding-based retrieval, context assembly, and generative summarization.

```
User Symptom Query
        │
        ▼
┌────────────────────────────┐
│  Sentence Embedding        │
│  all-MiniLM-L6-v2          │
│  384-dimensional vector    │
│  Mean Pooling + L2 Norm    │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  FAISS Vector Search       │
│  L2 Distance Index         │
│  Top-K=10 retrieval        │
│  Score Threshold ≤ 2.0     │
└────────────┬───────────────┘
             │
             ▼
  ┌──────────┴──────────┐
  │ Hybrid Mode?        │
  │ (classified_disease │
  │  provided?)         │
  └──────────┬──────────┘
      Yes ───┤───── No
      │             │
      ▼             ▼
┌────────────┐  ┌────────────┐
│  Targeted  │  │  General   │
│  Retrieval │  │  Retrieval │
│  (5 chunks)│  │  (10 chunks│
│  + General │  │   only)    │
│  (10 merge)│  │            │
└─────┬──────┘  └─────┬──────┘
      │               │
      └───────┬───────┘
              │
              ▼
┌────────────────────────────┐
│  Context Assembly          │
│  Top-5 chunks, 800 chars   │
│  Max 5 diseases in prompt  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  FLAN-T5-Base Generation   │
│  Beam Search (4 beams)     │
│  Max Output: 512 tokens    │
│  Length Penalty: 1.2       │
│  No-Repeat N-gram: 3      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Response Sanitization     │
│  40+ Regex Rules           │
│  Section Parsing           │
│  KB Enrichment             │
│  Medical Disclaimer        │
└────────────────────────────┘
```

### 7.2 Embedding and Retrieval

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`

| Parameter | Value |
|-----------|-------|
| Output Dimension | 384 |
| Pooling Strategy | Custom mean pooling with attention mask |
| Normalization | L2 normalization |
| Max Query Tokens | 256 |

**FAISS Index Configuration:**

| Parameter | Value |
|-----------|-------|
| Index Type | Flat L2 (exhaustive search) |
| Similarity Metric | L2 (Euclidean) distance — lower = more similar |
| Score Threshold | 2.0 (max L2 distance accepted) |
| Default Top-K | 10 |
| Metadata Format | `metadata.pkl` — list of `{disease, content}` dicts |

**Retrieval Functions:**

- `retrieve(query, top_k)` — General semantic search across all knowledge
- `retrieve_for_disease(query, disease, top_k)` — Disease-filtered retrieval
- `retrieve_all_for_disease(disease)` — All chunks for a given disease

**Hybrid Retrieval Strategy:**

When ClinicalBERT provides a classified disease, the retriever performs:
1. **Targeted retrieval:** 5 chunks filtered for the classified disease
2. **General retrieval:** 10 chunks across all diseases
3. **Merge and deduplicate:** Targeted chunks first, then unique general chunks (deduplication by first 100 characters), capped at 10 total

### 7.3 Generative Model

**Model:** `google/flan-t5-base` (~220M parameters, encoder-decoder)

| Parameter | Value |
|-----------|-------|
| Max Input Tokens | 1,024 |
| Max Output Tokens | 512 |
| Generation Strategy | Beam Search |
| Number of Beams | 4 |
| Temperature | 0.7 |
| Top-P (Nucleus) | 0.9 |
| Do Sample | False (deterministic beam search) |
| Length Penalty | 1.2 |
| No Repeat N-gram Size | 3 |
| Early Stopping | True |

**Prompt Template:**

```
You are a medical information assistant. Based on the patient's symptoms
and retrieved medical knowledge, provide a comprehensive yet concise
medical explanation.

Patient's symptoms: {query}

{disease_context}

Retrieved medical knowledge:
{formatted_chunks}

Please provide:
1. Most likely condition and brief explanation
2. Common causes of this condition
3. Key symptoms to watch for
4. Recommended next steps
5. When to seek immediate medical attention
6. General lifestyle and preventive advice

Important: This is for informational purposes only.
Always recommend consulting a healthcare professional for proper diagnosis.
```

### 7.4 Response Processing

Generated responses undergo **40+ regex-based sanitization rules** to clean model artifacts:

- Remove numbered prefixes and redundant labels
- Normalize whitespace, fix broken sentences
- Remove T5 generation artifacts (e.g., "Answer:", "Output:")
- Clean special characters and formatting inconsistencies

**Section Parsing** maps generated text into structured sections using keyword detection:

| Section Key | Keywords Matched |
|-------------|-----------------|
| Overview | "overview", "condition", "description", "about" |
| Symptoms | "symptom", "sign", "manifestation" |
| Causes | "cause", "etiology", "risk factor" |
| Treatment | "treatment", "medication", "therapy" |
| Complications | "complication", "severity", "danger" |
| Recommendations | "recommend", "lifestyle", "prevention", "advice" |
| Emergency | "emergency", "immediate", "urgent", "seek medical" |

### 7.5 Response Caching

| Parameter | Value |
|-----------|-------|
| Cache Strategy | In-memory dictionary with SHA-256 key hashing |
| Max Cache Entries | 200 (LRU eviction) |
| TTL | 300 seconds (5 minutes) |
| Cache Key Components | query text + classified disease |

---

## 8. Symptom Verification Mechanism

### 8.1 Two-Pass Verification Architecture

After ClinicalBERT produces initial predictions, a symptom verification layer re-scores each prediction by comparing reported symptoms against the knowledge base.

**Algorithm 1: Symptom Verification**

```
Input: predictions P = {(disease_i, confidence_i)}, user_text T
Output: re-ranked predictions P'

1. For each (disease, confidence) ∈ P:
2.     S_kb ← GetKnowledgeBaseSymptoms(disease)
3.     score_word ← WordOverlapScore(T, S_kb)          // Weight: 0.25
4.     score_phrase ← PhraseMatchScore(T, S_kb)         // Weight: 0.35
5.     score_keyword ← KeywordMatchScore(T, S_kb)       // Weight: 0.40
6.     symptom_score ← 0.25·score_word + 0.35·score_phrase + 0.40·score_keyword
7.     combined ← 0.60·confidence + 0.40·symptom_score
8.     If symptom_score < 0.15: flag as weak_match
9.     P' ← P'.append((disease, combined, symptom_score, weak_match))
10. Return sorted(P', key=combined, descending)
```

### 8.2 Multi-Component Scoring

| Component | Weight | Description |
|-----------|--------|-------------|
| Word Overlap | 25% | Jaccard similarity between tokenized user text and KB symptom terms |
| Phrase Matching | 35% | Multi-word phrase detection from knowledge base entries |
| Keyword Matching | 40% | Domain-specific medical keyword identification |

### 8.3 Synonym Expansion

The verifier maintains **26 synonym groups** for symptom normalization:

| Canonical Term | Synonyms |
|---------------|----------|
| fever | high temperature, pyrexia, febrile |
| headache | cephalalgia, head pain, cranial pain |
| nausea | feeling sick, queasiness, stomach upset |
| fatigue | tiredness, exhaustion, lethargy, malaise |
| rash | skin eruption, dermatitis, exanthem |
| ... | *(26 total groups)* |

### 8.4 Combined Score Fusion

$$S_{combined} = 0.60 \times C_{model} + 0.40 \times S_{symptom}$$

Where:
- $C_{model}$ = ClinicalBERT softmax confidence (0–1)
- $S_{symptom}$ = Weighted symptom verification score (0–1)
- Weak match threshold: $S_{symptom} < 0.15$

---

## 9. Emergency Detection System

### 9.1 Emergency Keyword Detection

The system implements real-time emergency detection scanning for 10 critical symptom keywords:

| # | Emergency Keyword | Associated Conditions |
|---|-------------------|----------------------|
| 1 | chest pain | Cardiac emergency |
| 2 | difficulty breathing | Respiratory failure |
| 3 | unconscious | Neurological emergency |
| 4 | severe bleeding | Hemorrhagic emergency |
| 5 | seizure | Neurological emergency |
| 6 | stroke | Cerebrovascular accident |
| 7 | heart attack | Myocardial infarction |
| 8 | anaphylaxis | Severe allergic reaction |
| 9 | poisoning | Toxicological emergency |
| 10 | suicidal | Mental health crisis |

### 9.2 Emergency Response Protocol

When emergency keywords are detected:

1. **Flag is_emergency = True** in response
2. **Return explanation** describing the emergency nature
3. **Provide helplines:**
   - Emergency Services: 112 / 911
   - Ambulance: 108
   - Poison Control: 1800-11-6117
4. **Urgent action message** prompting immediate medical attention
5. **List triggered keywords** for transparency

---

## 10. Report Summarization Pipeline

### 10.1 Multi-Phase Summarization Architecture

```
Uploaded File (PDF / Image / TXT)
        │
        ▼
┌────────────────────────────────────────┐
│  Phase 0: Text Extraction              │
│  pdfplumber → PyPDF2 (PDF)             │
│  EasyOCR → Pytesseract (Image)         │
│  UTF-8 decode (TXT)                    │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│  Phase 1a: SciSpacy Biomedical NER     │
│  en_core_sci_md model                  │
│  → Diseases, Medications, Symptoms,    │
│    Tests, Procedures, Anatomy          │
│  Drug suffix matching (28 patterns)    │
│  Noise filtering (60+ blacklist)       │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│  Phase 1b: Regex Structured Extraction │
│  → Patient demographics               │
│  → Lab values (30+ patterns) with      │
│    clinical interpretation             │
│  → Medication schedules               │
│  → Diagnostic findings                │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│  Phase 2: T5 Multi-Pass QA             │
│  (Optional gap-filling)               │
│  5 focused clinical questions          │
│  FLAN-T5 generates targeted answers    │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│  Merge & Deduplicate                   │
│  Combine NER + Regex + T5 findings     │
│  Remove redundant entities             │
│  Structured output assembly            │
└────────────────────────────────────────┘
```

### 10.2 Lab Value Interpretation

The system includes **30+ regex patterns** for automatic lab value extraction and clinical interpretation:

| Lab Test | Pattern Example | Interpretation Logic |
|----------|----------------|---------------------|
| Hemoglobin | `hemoglobin\s*[:=]?\s*(\d+\.?\d*)` | Low (<12 g/dL), Normal, High (>17 g/dL) |
| Blood Glucose | `glucose\s*[:=]?\s*(\d+\.?\d*)` | Low (<70), Normal, Pre-diabetic (100-125), High (>125) |
| WBC Count | `wbc\s*[:=]?\s*(\d+\.?\d*)` | Low (<4,500), Normal, High (>11,000) |
| Creatinine | `creatinine\s*[:=]?\s*(\d+\.?\d*)` | Normal (0.7-1.3), Elevated |
| Cholesterol | `cholesterol\s*[:=]?\s*(\d+\.?\d*)` | Desirable (<200), Borderline, High |
| HbA1c | `hba1c\s*[:=]?\s*(\d+\.?\d*)` | Normal (<5.7), Pre-diabetic, Diabetic (>6.5) |
| TSH | `tsh\s*[:=]?\s*(\d+\.?\d*)` | Hypo (<0.4), Normal, Hyper (>4.0) |

---

## 11. Query Processing Workflow

### 11.1 End-to-End Query Pipeline

**Algorithm 2: MedAI Query Processing**

```
Input: user_text T, uploaded_file F (optional)
Output: structured medical response R

 1. T ← sanitize_input(T)
 2. If F ≠ null:
 3.     report_text ← extract_text_from_file(F)
 4.     summary ← summarize_report(report_text)
 5.     Return format_summary_response(summary)
 6. 
 7. If is_greeting(T):
 8.     Return greeting_response()
 9. 
10. emergency ← detect_emergency(T)
11. If emergency.is_emergency:
12.     R.emergency ← emergency
13. 
14. predictions ← ClinicalBERT.predict(T, top_k=3)
15.     // Returns: [(disease, confidence, risk_level), ...]
16. 
17. For each (disease, confidence) ∈ predictions:
18.     verified ← symptom_verifier.verify(T, disease, confidence)
19.     predictions[i] ← verified  // re-ranked
20. 
21. If RAG_ENABLED:
22.     rag_result ← rag_pipeline.query(
23.         query = T,
24.         classified_disease = predictions[0].disease,
25.         all_predictions = predictions,
26.         use_cache = True
27.     )
28.     R.rag_response ← rag_result.rag_response
29.     R.retrieved_chunks ← rag_result.retrieved_chunks
30.     R.generation_meta ← rag_result.generation_meta
31. 
32. R.predictions ← enrich_with_disease_info(predictions)
33. R.disclaimer ← MEDICAL_DISCLAIMER
34. Return R
```

### 11.2 Greeting Detection

The system uses 15+ regex patterns to distinguish conversational inputs from medical queries:

```python
GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening))[\s!.,]*$",
    r"^(what'?s?\s*up|howdy|yo|sup)[\s!.,]*$",
    r"^(how\s*are\s*you|how\s*do\s*you\s*do)[\s!?.,]*$",
    r"^(thanks?|thank\s*you|thx)[\s!.,]*$",
    # ... 15+ patterns total
]

MEDICAL_SIGNAL_WORDS = [
    "pain", "fever", "ache", "sore", "hurt", "symptom",
    "disease", "condition", "diagnos", "medication", ...
]
```

If text matches a greeting pattern AND contains no medical signal words, it is routed to a conversational response instead of medical analysis.

---

## 12. Training Methodology

### 12.1 ClinicalBERT Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Base Model | ClinicalBERT (pre-trained on MIMIC-III clinical notes) |
| Fine-Tuning Task | Single-label 22-class classification |
| Loss Function | CrossEntropyLoss |
| Input Max Length | 128 tokens |
| Train/Test Split | 80/20 (stratified by disease class) |
| Random Seed | 42 |
| Output Head | Linear(768 → 22) + Softmax |
| Evaluation Batch Size | 16 |
| Device | Auto-detect CUDA / CPU |

### 12.2 Transfer Learning Strategy

The model leverages transfer learning from ClinicalBERT's pre-training on clinical notes (MIMIC-III database), which provides:

- Clinical vocabulary understanding (28,996 WordPiece tokens)
- Medical term disambiguation
- Symptom–disease semantic relationships
- Clinical note structure awareness

Fine-tuning adapts these representations to the specific task of mapping conversational symptom descriptions to 22 disease categories.

---

## 13. Experimental Setup

### 13.1 Evaluation Protocol

| Parameter | Value |
|-----------|-------|
| Test Set Size | 20% of dataset (stratified) |
| Batch Size | 16 |
| Metrics | Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Top-3 Accuracy |
| Visualization | Confusion matrix heatmap |
| Inference Timing | Per-sample latency (ms) |

### 13.2 Baseline Models

Four traditional ML models serve as baselines, all using TF-IDF vectorization (10,000 features, bigram range):

| Model | Key Hyperparameters |
|-------|-------------------|
| Naive Bayes | α = 1.0 (Laplace smoothing) |
| SVM (Linear) | max_iter = 5,000 |
| Random Forest | n_estimators = 200, max_depth = unlimited |
| Logistic Regression | max_iter = 2,000 |

### 13.3 Evaluation Metrics

The following metrics are computed for all models:

**Accuracy:**
$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}$$

**Macro Precision (averaged across all classes):**
$$\text{Precision}_{macro} = \frac{1}{N} \sum_{i=1}^{N} \frac{TP_i}{TP_i + FP_i}$$

**Macro Recall:**
$$\text{Recall}_{macro} = \frac{1}{N} \sum_{i=1}^{N} \frac{TP_i}{TP_i + FN_i}$$

**Macro F1-Score:**
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Top-3 Accuracy:**
$$\text{Top\text{-}3 Acc} = \frac{|\{x : y_{true} \in \text{top-3}(f(x))\}|}{|X_{test}|}$$

---

## 14. Experimental Results

### 14.1 Classification Performance

| Model | Accuracy | Macro F1 | Weighted F1 | Training Time | Inference Time |
|-------|----------|----------|-------------|---------------|----------------|
| Naive Bayes + TF-IDF | — | — | — | Fast | Fast |
| SVM (Linear) + TF-IDF | — | — | — | Moderate | Fast |
| Random Forest + TF-IDF | — | — | — | Moderate | Fast |
| Logistic Regression + TF-IDF | — | — | — | Moderate | Fast |
| **ClinicalBERT (Proposed)** | **80.6%** | — | — | Slow | ~10–20ms/sample |

*Note: Baseline figures are computed by `baseline_comparison.py`. Fill in exact values after running evaluation.*

### 14.2 RAG Pipeline Performance

| Metric | Value |
|--------|-------|
| ClinicalBERT Inference | ~10–20 ms/sample (GPU) |
| FAISS Retrieval (Top-10) | ~50 ms |
| FLAN-T5 Generation | ~500–1000 ms/query |
| Total Pipeline Latency | ~600–1100 ms end-to-end |
| Cache Hit Latency | < 5 ms |

### 14.3 System Health Diagnostics

The `/api/health/` endpoint reports real-time system status:

```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX ...",
  "models_loaded": {
    "clinicalbert": true,
    "embedding_model": true,
    "generator_model": true,
    "faiss_index": true,
    "ner_model": true
  }
}
```

---

## 15. Frontend Design and User Experience

### 15.1 User Interface Architecture

The frontend is built with **React 18** and follows a component-based architecture:

| Component | Purpose |
|-----------|---------|
| **ChatPage** | Main interaction interface with conversation management |
| **Sidebar** | ChatGPT-style dark sidebar with conversation history, search, and time-based grouping (Today, Yesterday, Previous 7 Days, Older) |
| **ChatBox** | Multi-line text input + file upload (PDF, TXT, PNG, JPG, JPEG — max 10 MB) |
| **ChatMessage** | Renders bot/user messages with type-specific formatting (prediction, summary, error, loading) |
| **ResultCard** | Disease prediction cards with confidence bars, symptoms chips, and risk color-coding |
| **History** | Tabular view of last 50 predictions with color-coded risk levels |
| **LoginPage** | Split-layout (branding + form) with gradient background |
| **RegisterPage** | Multi-field registration with password strength indicator (Weak/Fair/Good/Strong) |
| **ProfilePage** | Editable health profile with BMI calculation, medical history, emergency contact |

### 15.2 Conversation Management

- **Persistence:** localStorage-based multi-conversation persistence (`medai_conversations`)
- **Structure:** Each conversation contains `{id, title, messages[], createdAt, updatedAt}`
- **Auto-titling:** First user message truncated to 36 characters
- **Message Types:** `text`, `prediction`, `summary`, `error`, `loading`

### 15.3 Authentication Flow

```
User Registration
    │
    ├─→ POST /api/auth/register/
    │      Body: {username, email, password, first_name, last_name}
    │      Response: {token, user, profile}
    │
    ├─→ Token stored in localStorage (medai_token)
    │
    └─→ Axios interceptor adds "Authorization: Token <token>" to all requests
    
User Login
    │
    ├─→ POST /api/auth/login/
    │      Body: {username, password}
    │      Response: {token, user, profile}
    │
    └─→ Profile validated on mount; 401 clears stale tokens
```

### 15.4 Prediction Response Display

The chat interface renders disease predictions with:

- **Empathetic introductions:** 4 randomly selected greeting sentences
- **Risk-level color coding:** Red (high), Amber (moderate), Blue (low)
- **Knowledge base sections** with icons: 📋 Overview, 🩹 Symptoms, 🔬 Causes, 💊 Treatment, ⚠️ Complications, 🛡️ Prevention
- **Emergency alerts:** Yellow alert box with helpline information
- **Medical disclaimer:** Present on every prediction response

---

## 16. Database Schema

### 16.1 Entity-Relationship Model

```
┌──────────────────┐     1:1     ┌─────────────────────┐
│   AUTH_USER       │────────────│   API_USERPROFILE    │
├──────────────────┤             ├─────────────────────┤
│ id (PK)          │             │ id (PK)             │
│ username         │             │ user_id (FK)        │
│ email            │             │ phone               │
│ password (hash)  │             │ date_of_birth       │
│ first_name       │             │ gender              │
│ last_name        │             │ blood_group         │
│ date_joined      │             │ height_cm           │
│ is_active        │             │ weight_kg           │
└──────────────────┘             │ allergies           │
                                 │ medical_conditions  │
                                 │ emergency_contact   │
                                 │ address             │
                                 └─────────────────────┘

┌──────────────────────┐
│   API_PREDICTIONLOG  │
├──────────────────────┤
│ id (PK)              │
│ user_id (FK)         │
│ symptoms             │
│ predicted_disease     │
│ confidence           │
│ risk_level           │
│ is_emergency         │
│ created_at           │
└──────────────────────┘
```

---

## 17. Security and Safety Measures

### 17.1 Input Validation

- Text length enforcement (3–5,000 characters)
- File type whitelist (PDF, TXT, PNG, JPG, JPEG)
- File size limit (10 MB)
- Input sanitization against injection attacks

### 17.2 Authentication Security

- Token-based authentication (DRF TokenAuthentication)
- Password hashing (Django's PBKDF2 with SHA256)
- Session-based CSRF protection
- CORS configured for localhost:3000 only
- Stale token detection and automatic cleanup (401 response interceptor)

### 17.3 Medical Safety

- **Disclaimer on every response:** "This system provides informational guidance only. It is NOT a medical diagnosis."
- **Emergency escalation:** Automatic detection of 10 critical keywords with helpline routing
- **No prescription recommendations:** System provides general advice only
- **Confidence transparency:** All predictions include confidence percentages and risk levels

---

## 18. System Diagrams

### 18.1 End-to-End ML Pipeline

```mermaid
flowchart TD
    A[User Input: Symptoms / Report] --> B{Input Type?}
    B -->|Text| C[Sanitize Input]
    B -->|File| D[Extract Text<br/>PDF/Image/TXT]
    
    D --> E[Report Summarizer<br/>NER + Regex + T5 QA]
    E --> F[Structured Summary<br/>Response]
    
    C --> G{Is Greeting?}
    G -->|Yes| H[Greeting Response]
    G -->|No| I[Emergency Detection<br/>10 Keywords]
    
    I --> J[ClinicalBERT<br/>Classification]
    J --> K[Top-3 Predictions<br/>+ Confidence Scores]
    
    K --> L[Symptom Verifier<br/>Word + Phrase + Keyword<br/>60% Model + 40% Match]
    L --> M[Re-ranked Predictions]
    
    M --> N{RAG Enabled?}
    N -->|Yes| O[FAISS Retrieval<br/>Hybrid: Targeted + General]
    N -->|No| P[Return Classification Only]
    
    O --> Q[Context Assembly<br/>Top-5 Chunks, 800 chars]
    Q --> R[FLAN-T5 Generation<br/>4-beam search]
    R --> S[Response Sanitization<br/>40+ Regex Rules]
    S --> T[Section Parsing +<br/>KB Enrichment]
    T --> U[Final Multi-Disease<br/>Medical Report]
    
    U --> V[Add Disclaimer +<br/>Emergency Info]
    V --> W[JSON Response<br/>to Frontend]
    
    style J fill:#4361ee,color:#fff
    style O fill:#7209b7,color:#fff
    style R fill:#f72585,color:#fff
    style L fill:#3a0ca3,color:#fff
```

### 18.2 ClinicalBERT Architecture Diagram

```mermaid
flowchart TD
    A[Input Text] --> B[WordPiece Tokenizer<br/>vocab=28,996<br/>max_len=128]
    B --> C[Token Embeddings +<br/>Position Embeddings +<br/>Segment Embeddings]
    
    C --> D[Transformer Layer 1<br/>12 Attention Heads<br/>768 Hidden Dim]
    D --> E[Transformer Layer 2]
    E --> F[...]
    F --> G[Transformer Layer 12]
    
    G --> H["[CLS] Token<br/>768-dim Vector"]
    H --> I[Dropout 0.1]
    I --> J[Dense Layer<br/>768 → 22]
    J --> K[Softmax]
    K --> L[Top-K Predictions]
    
    L --> M["Disease 1: 72.3%<br/>Disease 2: 15.1%<br/>Disease 3: 8.4%"]
    
    style A fill:#e8f5e9
    style K fill:#4361ee,color:#fff
    style M fill:#f3e5f5
```

### 18.3 RAG Pipeline Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant API as Django API
    participant CB as ClinicalBERT
    participant SV as Symptom Verifier
    participant EM as Embedding Model
    participant FI as FAISS Index
    participant T5 as FLAN-T5
    participant KB as Knowledge Base
    participant CA as Cache
    
    U->>API: POST /predict-rag/ {symptoms}
    API->>API: Sanitize + Greeting Check
    API->>CB: predict(text, top_k=3)
    CB-->>API: [(disease, confidence), ...]
    API->>SV: verify(text, disease, confidence)
    SV->>KB: Get disease symptoms
    SV-->>API: Re-ranked predictions
    
    API->>CA: Check cache (SHA-256 key)
    alt Cache Hit
        CA-->>API: Cached response
    else Cache Miss
        API->>EM: Encode query (384-dim)
        EM->>FI: L2 search (top-10)
        FI-->>EM: Retrieved chunks
        EM-->>API: Scored chunks
        API->>T5: generate(query, chunks, disease)
        T5-->>API: Generated explanation
        API->>API: Sanitize + Format
        API->>CA: Store in cache (TTL=300s)
    end
    
    API-->>U: JSON {rag_response, predictions, meta}
```

### 18.4 System Architecture Diagram

```mermaid
graph TB
    subgraph Frontend["Frontend (React 18)"]
        LP[LoginPage]
        RP[RegisterPage]
        CP[ChatPage]
        PP[ProfilePage]
        SB[Sidebar]
        CB[ChatBox]
        CM[ChatMessage]
        RC[ResultCard]
        HI[History]
    end
    
    subgraph API["API Layer (Django REST Framework)"]
        PR["/predict-rag/"]
        PC["/predict/"]
        HL["/health/"]
        HY["/history/"]
        AU["/auth/*"]
    end
    
    subgraph ML["ML Inference Layer"]
        BERT[ClinicalBERT<br/>110M params]
        RAG[RAG Pipeline]
        SUM[Report Summarizer]
        NER[Medical NER<br/>SciSpacy]
        VER[Symptom Verifier]
        EMG[Emergency Detector]
    end
    
    subgraph RAGDetail["RAG Components"]
        EMB[MiniLM-L6-v2<br/>384-dim Embeddings]
        FAI[FAISS Index<br/>L2 Distance]
        GEN[FLAN-T5-Base<br/>220M params]
    end
    
    subgraph Data["Data Layer"]
        DB[(MySQL<br/>medai_db)]
        KBD[Knowledge Base<br/>22 Disease JSONs]
        DI[disease_info.json]
        MW[Model Weights]
    end
    
    Frontend -->|Axios HTTP| API
    PR --> ML
    PC --> BERT
    RAG --> RAGDetail
    ML --> Data
    
    style BERT fill:#4361ee,color:#fff
    style GEN fill:#f72585,color:#fff
    style FAI fill:#7209b7,color:#fff
    style EMB fill:#3a0ca3,color:#fff
```

---

## 19. Conclusion

This paper presents MedAI, a comprehensive AI-powered healthcare chatbot that advances the state of the art in symptom-based disease prediction by integrating multiple NLP and ML paradigms into a unified system. The key findings and contributions are:

1. **Hybrid Classification-RAG Architecture:** The combination of ClinicalBERT's classification precision with FAISS-based retrieval and FLAN-T5 generation produces responses that are both accurate and contextually grounded, mitigating the hallucination risks associated with unconstrained generative models.

2. **Symptom Verification Layer:** The two-pass verification mechanism with weighted score fusion (60% model confidence + 40% symptom match) using Jaccard similarity, phrase matching, and synonym expansion provides a validation layer that catches predictions misaligned with reported symptoms.

3. **Multi-Modal Report Processing:** The three-phase summarization pipeline (SciSpacy NER + regex extraction + T5 QA) demonstrates a practical approach to automated medical report understanding across PDF, image, and text formats.

4. **Production Readiness:** The system's architecture — with thread-safe model loading, response caching (SHA-256 keyed, 5-minute TTL), graceful degradation (fallback chains for PDF/OCR), and comprehensive error handling — demonstrates that research prototypes can be engineered for production deployment.

5. **Safety-First Design:** Emergency detection with helpline routing, mandatory medical disclaimers, and confidence transparency ensure the system operates within appropriate ethical boundaries for a non-diagnostic advisory tool.

### 19.1 Limitations

- The system is trained on 22 disease categories, which represents a focused but limited subset of the medical condition space.
- ClinicalBERT's max sequence length of 128 tokens may truncate longer, more detailed symptom descriptions.
- The knowledge base relies on curated JSON content that requires manual updates as medical knowledge evolves.
- FLAN-T5-Base, while efficient, has limited generative capacity compared to larger LLMs.

### 19.2 Future Work

- Expand disease coverage beyond 22 categories with hierarchical multi-label classification.
- Integrate larger generative models (e.g., FLAN-T5-Large, Llama-3) for richer medical explanations.
- Implement continuous learning from user feedback to refine predictions over time.
- Add multi-language support for broader accessibility.
- Incorporate clinical guidelines databases (e.g., UpToDate, DynaMed) for evidence-based recommendations.
- Deploy as a HIPAA-compliant cloud service with end-to-end encryption.

---

## References

1. Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
2. Huang, K., Altosaar, J., & Ranganath, R. (2019). ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission. *arXiv:1904.05342*.
3. Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
4. Chung, H.W., Hou, L., Longpre, S., et al. (2022). Scaling Instruction-Finetuned Language Models (FLAN-T5). *arXiv:2210.11416*.
5. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-Scale Similarity Search with GPUs. *IEEE Transactions on Big Data*.
6. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
7. Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing. *BioNLP Workshop, ACL*.
8. Singhal, K., Azizi, S., Tu, T., et al. (2023). Large Language Models Encode Clinical Knowledge. *Nature*.
9. Li, Y., Li, Z., Zhang, K., et al. (2023). ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA). *arXiv:2303.14070*.
10. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS*.
11. Lee, J., Yoon, W., Kim, S., et al. (2020). BioBERT: A Pre-trained Biomedical Language Representation Model. *Bioinformatics*.
12. Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *EMNLP (Demo)*.

---

## Appendix A: Algorithm Pseudocode

### A.1 ClinicalBERT Inference

```
FUNCTION predict(text, top_k=3):
    model, tokenizer ← load_model_and_tokenizer()
    inputs ← tokenizer.encode(text, max_length=128, padding=True, truncation=True)
    
    WITH torch.no_grad():
        logits ← model(inputs).logits
    
    probabilities ← softmax(logits, dim=-1)
    top_values, top_indices ← torch.topk(probabilities, k=top_k)
    
    results ← []
    FOR i IN range(top_k):
        disease ← LABEL_MAP[top_indices[i]]
        confidence ← top_values[i] × 100
        risk ← classify_risk(confidence)
        results.append({disease, confidence, risk})
    
    RETURN results
```

### A.2 FAISS Hybrid Retrieval

```
FUNCTION hybrid_retrieve(query, classified_disease, top_k=10):
    query_embedding ← encode(query)  // 384-dim, L2 normalized
    
    // Targeted retrieval for classified disease
    targeted_chunks ← []
    FOR chunk IN faiss_search(query_embedding, k=top_k):
        IF chunk.disease == classified_disease:
            targeted_chunks.append(chunk)
        IF len(targeted_chunks) >= 5:
            BREAK
    
    // General retrieval
    general_chunks ← faiss_search(query_embedding, k=10)
    
    // Merge with deduplication
    merged ← targeted_chunks
    seen ← {chunk.content[:100] FOR chunk IN targeted_chunks}
    FOR chunk IN general_chunks:
        IF chunk.content[:100] NOT IN seen AND len(merged) < 10:
            merged.append(chunk)
            seen.add(chunk.content[:100])
    
    RETURN merged
```

### A.3 Response Generation

```
FUNCTION generate(query, chunks, classified_disease):
    prompt ← build_prompt(query, chunks, classified_disease)
        // Template with 6-point medical structure
        // Top-5 chunks, 800 chars each, max 5 diseases
    
    inputs ← tokenizer.encode(prompt, max_length=1024)
    
    output ← model.generate(
        inputs,
        max_new_tokens = 512,
        num_beams = 4,
        length_penalty = 1.2,
        no_repeat_ngram_size = 3,
        early_stopping = True
    )
    
    generated_text ← tokenizer.decode(output)
    RETURN sanitize(generated_text)  // 40+ regex cleanup rules
```

---

## Appendix B: Configuration Parameters

| Category | Parameter | Value |
|----------|-----------|-------|
| **ClinicalBERT** | Model Path | `ml_engine/clinicalbert-disease` |
| | Max Length | 128 tokens |
| | Precision | float32 |
| | Parameters | ~110M |
| **Embedding** | Model | `all-MiniLM-L6-v2` |
| | Dimension | 384 |
| | Max Query Tokens | 256 |
| **FAISS** | Index Path | `rag/faiss_index.bin` |
| | Metadata Path | `rag/metadata.pkl` |
| | Top-K | 10 |
| | Score Threshold | 2.0 L2 |
| **FLAN-T5** | Model | `google/flan-t5-base` |
| | Max Input | 1,024 tokens |
| | Max Output | 512 tokens |
| | Beams | 4 |
| | Temperature | 0.7 |
| | Top-P | 0.9 |
| | Length Penalty | 1.2 |
| **Cache** | TTL | 300 seconds |
| | Max Entries | 200 |
| **NER** | Model | `en_core_sci_md` |
| | Drug Patterns | 28 suffix patterns |
| | Blacklist Size | 60+ terms |
| **Verification** | Word Overlap Weight | 0.25 |
| | Phrase Match Weight | 0.35 |
| | Keyword Match Weight | 0.40 |
| | Model Confidence Weight | 0.60 |
| | Symptom Score Weight | 0.40 |
| | Weak Match Threshold | 0.15 |
| | Synonym Groups | 26 |

---

## Appendix C: API Response Schema

### C.1 Prediction Response (`POST /api/predict-rag/`)

```json
{
  "predictions": [
    {
      "disease": "dengue",
      "confidence": 72.3,
      "risk_level": "High Probability",
      "info": {
        "description": "Dengue is a mosquito-borne viral infection...",
        "common_symptoms": ["high fever", "severe headache", "body pain"],
        "advice": "Seek medical attention if symptoms persist",
        "knowledge_base": {
          "overview": "...",
          "symptoms": "...",
          "causes": "...",
          "treatment": "...",
          "when_to_see_a_doctor": "...",
          "preventions": "..."
        }
      }
    }
  ],
  "rag_response": "🏥 Likely Condition: Dengue (72.3%)\n\n...",
  "retrieved_chunks": [
    {
      "disease": "dengue",
      "score": 0.45,
      "rank": 1,
      "snippet": "Dengue is an acute febrile illness..."
    }
  ],
  "generation_meta": {
    "prompt_tokens": 856,
    "output_tokens": 412,
    "generation_latency_ms": 734.2
  },
  "pipeline_latency_ms": 892.5,
  "cache_hit": false,
  "emergency": {
    "is_emergency": false,
    "explanation": "",
    "helplines": "",
    "message": "",
    "triggered_keywords": []
  },
  "disclaimer": "This system provides informational guidance only..."
}
```

---

*Paper generated from MedAI project repository analysis. All technical details reverse-engineered from source code.*
