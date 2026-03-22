================================================================================
  MedAI — Architecture Diagrams for Research Paper
================================================================================
  
  These diagrams can be rendered using:
    - Mermaid Live Editor: https://mermaid.live
    - Draw.io (import Mermaid): https://app.diagrams.net
    - VS Code Mermaid Preview extension
  
  Copy each Mermaid code block into the renderer to get publication-ready
  PNG/SVG diagrams for your research paper.

================================================================================


================================================================================
DIAGRAM 1: SYSTEM ARCHITECTURE (4-Layer)
================================================================================

Use this for: Section 3.6 System Architecture / Figure 1

```mermaid
graph TB
    subgraph Frontend["Presentation Layer — React 18"]
        UI["Chat Interface<br/>(Multi-Conversation)"]
        Auth["Authentication<br/>(Login / Register)"]
        Profile["User Profile<br/>(Health Info / BMI)"]
        Upload["File Upload<br/>(PDF / Image / TXT)"]
    end

    subgraph API["API Layer — Django REST Framework"]
        Predict["/api/predict-rag/<br/>Disease Prediction"]
        Health["/api/health/<br/>System Health"]
        History["/api/history/<br/>Prediction Logs"]
        AuthAPI["/api/auth/*<br/>Token Auth"]
    end

    subgraph ML["ML Inference Layer"]
        Greeting["Greeting<br/>Detection"]
        Emergency["Emergency<br/>Detection"]
        BERT["ClinicalBERT<br/>(22-Class Classifier)"]
        Verifier["Symptom<br/>Verifier"]
        RAG["RAG Pipeline"]
        Summarizer["Report<br/>Summarizer"]
    end

    subgraph RAGDetail["RAG Pipeline Components"]
        Embed["MiniLM-L6-v2<br/>Embeddings (384d)"]
        FAISS["FAISS<br/>Vector Search"]
        T5["FLAN-T5-Base<br/>Generation"]
    end

    subgraph NER["Medical NER"]
        SciSpacy["SciSpacy<br/>Entity Extraction"]
    end

    subgraph Data["Data Layer"]
        MySQL["MySQL 8.x<br/>(Users, Logs)"]
        KB["Knowledge Base<br/>(22 Disease JSONs)"]
        VectorDB["FAISS Index<br/>(Pre-built Vectors)"]
        Model["ClinicalBERT<br/>Weights (.safetensors)"]
    end

    UI -->|Axios HTTP| Predict
    Auth -->|POST| AuthAPI
    Profile -->|GET/PUT| AuthAPI
    Upload -->|FormData| Predict

    Predict --> Greeting
    Predict --> Emergency
    Predict --> BERT
    Predict --> Summarizer
    Health --> ML
    History --> MySQL

    BERT --> Verifier
    Verifier --> RAG
    RAG --> Embed
    Embed --> FAISS
    FAISS --> T5

    Summarizer --> SciSpacy

    BERT -.->|loads| Model
    Verifier -.->|reads| KB
    FAISS -.->|searches| VectorDB
    AuthAPI -.->|queries| MySQL
    Predict -.->|logs| MySQL
```


================================================================================
DIAGRAM 2: ML PIPELINE FLOWCHART (End-to-End)
================================================================================

Use this for: Section 3 Methodology / Figure 2

```mermaid
flowchart TD
    Start(["User Input — Text / File Upload"])

    Start --> Sanitize["Input Sanitization<br/>(3-5000 chars, UTF-8)"]

    Sanitize --> FileCheck{File<br/>Uploaded?}
    FileCheck -->|Yes| Extract["Text Extraction<br/>PDF: pdfplumber<br/>Image: EasyOCR<br/>TXT: UTF-8"]
    FileCheck -->|No| Combine
    Extract --> Combine["Combine Text + File Content"]

    Combine --> GreetCheck{Greeting<br/>Detected?}
    GreetCheck -->|Yes| GreetResp["Conversational Response"]

    GreetCheck -->|No| ReportCheck{Report Summarization<br/>Intent?}
    ReportCheck -->|Yes| ReportPipe["Report Summarization Pipeline<br/>Pass 1: FLAN-T5 QA<br/>Pass 2: Regex Extraction<br/>Pass 3: SciSpacy NER"]

    ReportCheck -->|No| EmergCheck["Emergency Detection<br/>(12 Critical Keywords)"]

    EmergCheck --> ClinBERT["ClinicalBERT Classification<br/>(Top-3 Predictions)"]

    ClinBERT --> Verify["Symptom Verification<br/>60% Model + 40% Symptom Match<br/>Synonym Expansion + Jaccard"]

    Verify --> FAISSSearch["FAISS Semantic Retrieval<br/>Top-10 Chunks (384d MiniLM)"]

    FAISSSearch --> KBLookup["Knowledge Base Assembly<br/>(Overview, Symptoms,<br/>Causes, Treatment)"]

    KBLookup --> T5Gen["FLAN-T5 Generation<br/>Context-Grounded Explanation"]

    T5Gen --> Cache["Response Cache<br/>(SHA-256 Key, TTL 300s)"]

    Cache --> Response["Final Response<br/>+ Medical Disclaimer<br/>+ Emergency Alert (if any)"]

    style Start fill:#4CAF50,color:#fff
    style ClinBERT fill:#2196F3,color:#fff
    style T5Gen fill:#FF9800,color:#fff
    style FAISSSearch fill:#9C27B0,color:#fff
    style EmergCheck fill:#f44336,color:#fff
    style Response fill:#4CAF50,color:#fff
```


================================================================================
DIAGRAM 3: RAG PIPELINE DETAIL
================================================================================

Use this for: Section 3.5 Knowledge Retrieval / Figure 3

```mermaid
flowchart LR
    subgraph Input["Input"]
        Query["Symptom Query"]
    end

    subgraph Encoding["Encode"]
        MiniLM["all-MiniLM-L6-v2<br/>384-dim Vector"]
    end

    subgraph Retrieval["Retrieve"]
        FAISS["FAISS Index<br/>L2 Nearest Neighbor<br/>Top-K = 10"]
        Filter["Threshold Filter<br/>(distance ≤ 2.0)"]
    end

    subgraph Context["Assemble Context"]
        Chunks["Retrieved Chunks"]
        KB["Knowledge Base<br/>Disease Sections"]
        Prompt["Instruction Prompt<br/>Construction"]
    end

    subgraph Generation["Generate"]
        T5["FLAN-T5-Base<br/>Max Output: 512 tokens<br/>Temperature: 0.7"]
        Sanitize["Response Sanitization<br/>(Remove citations, URLs)"]
    end

    subgraph Output["Output"]
        Resp["Structured Response<br/>Overview | Symptoms<br/>Causes | Treatment<br/>Complications | Prevention"]
    end

    Query --> MiniLM
    MiniLM --> FAISS
    FAISS --> Filter
    Filter --> Chunks
    Chunks --> Prompt
    KB --> Prompt
    Prompt --> T5
    T5 --> Sanitize
    Sanitize --> Resp

    style MiniLM fill:#2196F3,color:#fff
    style FAISS fill:#9C27B0,color:#fff
    style T5 fill:#FF9800,color:#fff
    style Resp fill:#4CAF50,color:#fff
```


================================================================================
DIAGRAM 4: ClinicalBERT MODEL ARCHITECTURE
================================================================================

Use this for: Section 3.3 Symptom Classification Model / Figure 4

```mermaid
flowchart TD
    Input["Input: Symptom Text<br/>'I have fever, headache, body pain'"]
    
    Tokenizer["BertTokenizer<br/>(vocab: 28,996 tokens)"]
    
    Embed["Token + Position + Segment<br/>Embeddings (768-dim)"]
    
    subgraph Encoder["12x Transformer Encoder Layers"]
        Attn["Multi-Head Self-Attention<br/>(12 heads, 64 dim/head)"]
        FFN["Feed-Forward Network<br/>(768 → 3072 → 768)"]
        Norm["Layer Norm + Residual"]
    end
    
    CLS["[CLS] Token<br/>Representation (768-dim)"]
    
    Classifier["Linear Classification Head<br/>(768 → 22 classes)"]
    
    Softmax["Softmax<br/>Probability Distribution"]
    
    Output["Top-3 Predictions<br/>Disease + Confidence %"]

    Input --> Tokenizer
    Tokenizer --> Embed
    Embed --> Encoder
    Attn --> FFN
    FFN --> Norm
    Encoder --> CLS
    CLS --> Classifier
    Classifier --> Softmax
    Softmax --> Output

    style Input fill:#E8F5E9
    style Encoder fill:#E3F2FD
    style Classifier fill:#FFF3E0
    style Output fill:#E8F5E9
```


================================================================================
DIAGRAM 5: REPORT SUMMARIZATION PIPELINE
================================================================================

Use this for: Section 3.6 or a separate subsection / Figure 5

```mermaid
flowchart TD
    Upload["Medical Report Upload<br/>(PDF / Image / TXT)"]
    
    subgraph Extract["Text Extraction"]
        PDF["pdfplumber<br/>(+ PyPDF2 fallback)"]
        OCR["EasyOCR<br/>(+ pytesseract fallback)"]
        TXT["UTF-8 Decode"]
    end
    
    Intent["Intent Detection<br/>(summarize / explain / read)"]
    
    subgraph Pass1["Pass 1: T5 Question Answering"]
        Q1["Patient Details?"]
        Q2["Diagnoses?"]
        Q3["Test Results?"]
        Q4["Medications?"]
        Q5["Recommendations?"]
    end
    
    subgraph Pass2["Pass 2: Regex Extraction"]
        R1["Patient Info Patterns"]
        R2["Lab Value Patterns (40+)"]
        R3["Drug Patterns + Dosage"]
        R4["Diagnosis Section Headers"]
    end
    
    subgraph Pass3["Pass 3: SciSpacy NER"]
        E1["Diseases"]
        E2["Medications"]
        E3["Symptoms"]
        E4["Tests / Procedures"]
    end
    
    Merge["Merge & Deduplicate<br/>All Three Passes"]
    
    Output["Structured Summary<br/>Patient Info | Diagnoses<br/>Medications | Lab Values<br/>Recommendations | Entities"]

    Upload --> Extract
    PDF --> Intent
    OCR --> Intent
    TXT --> Intent
    Intent --> Pass1
    Intent --> Pass2
    Intent --> Pass3
    Pass1 --> Merge
    Pass2 --> Merge
    Pass3 --> Merge
    Merge --> Output

    style Upload fill:#E8F5E9
    style Pass1 fill:#E3F2FD
    style Pass2 fill:#FFF3E0
    style Pass3 fill:#FCE4EC
    style Output fill:#E8F5E9
```


================================================================================
DIAGRAM 6: DATABASE ER DIAGRAM
================================================================================

Use this for: Database Schema section / Figure 6

```mermaid
erDiagram
    AUTH_USER {
        int id PK
        varchar username UK
        varchar email
        varchar password
        varchar first_name
        varchar last_name
        boolean is_active
        datetime date_joined
    }

    API_USERPROFILE {
        int id PK
        int user_id FK
        varchar phone
        date date_of_birth
        varchar gender
        varchar blood_group
        decimal height_cm
        decimal weight_kg
        text allergies
        text medical_conditions
        varchar emergency_contact
        text address
        varchar avatar_initial
        datetime created_at
        datetime updated_at
    }

    API_PREDICTIONLOG {
        int id PK
        text symptoms
        varchar predicted_disease
        float confidence
        varchar risk_level
        boolean is_emergency
        datetime created_at
    }

    AUTH_USER ||--|| API_USERPROFILE : "has profile"
```


================================================================================
DIAGRAM 7: AUTHENTICATION FLOW
================================================================================

Use this for: System design section (optional) / Figure 7

```mermaid
sequenceDiagram
    participant U as User (Browser)
    participant F as React Frontend
    participant B as Django Backend
    participant DB as MySQL Database

    Note over U,DB: Registration Flow
    U->>F: Fill register form
    F->>B: POST /api/auth/register/
    B->>DB: Create User + UserProfile
    DB-->>B: User created
    B-->>F: {token, user} 200 OK
    F->>F: Store token in localStorage
    F-->>U: Redirect to chat

    Note over U,DB: Login Flow
    U->>F: Enter credentials
    F->>B: POST /api/auth/login/
    B->>DB: Validate credentials
    DB-->>B: User found
    B-->>F: {token, user} 200 OK
    F->>F: Store token in localStorage

    Note over U,DB: Authenticated Request
    U->>F: Enter symptoms
    F->>B: POST /api/predict-rag/<br/>Header: Token xxx
    B->>B: Validate token
    B->>B: ML Pipeline
    B->>DB: Log prediction
    B-->>F: {predictions, rag_response}
    F-->>U: Display results

    Note over U,DB: Logout
    U->>F: Click logout
    F->>B: POST /api/auth/logout/
    B->>DB: Delete token
    F->>F: Clear localStorage
    F-->>U: Redirect to login
```


================================================================================
HOW TO USE THESE DIAGRAMS
================================================================================

1. Go to https://mermaid.live
2. Paste any code block above (between ```mermaid and ```)
3. The diagram renders instantly
4. Click "Export" → PNG or SVG (use SVG for best quality)
5. Insert into your research paper as Figure 1, 2, 3, etc.

Alternative tools:
    - Draw.io: File → Import → Paste Mermaid code
    - VS Code: Install "Mermaid Preview" extension
    - LaTeX: Use the mermaid-js package or export as PDF

Figure captions suggested:
    - Fig. 1: System architecture of the proposed MedAI framework
    - Fig. 2: End-to-end ML pipeline flowchart
    - Fig. 3: RAG pipeline for knowledge-grounded response generation
    - Fig. 4: ClinicalBERT model architecture for disease classification
    - Fig. 5: Multi-pass medical report summarization pipeline
    - Fig. 6: Database entity-relationship diagram
    - Fig. 7: Authentication and request flow sequence diagram

================================================================================
