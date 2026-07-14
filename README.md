# Smart-Resume-Screener
Smart Resume Screener is an AI-powered application that analyzes and ranks resumes based on job descriptions using NLP and machine learning. It automates candidate screening, provides ATS-style matching scores, identifies key skills, and helps recruiters make faster hiring decisions.

```mermaid
flowchart LR

A[Upload Resumes PDF/DOCX]
B[Enter Job Description]

A --> C[Resume Parser]
B --> D[JD Analyzer]

C --> E[Text Extraction]
D --> F[Skill Extraction]

E --> G[Embedding Generation]
F --> G

G --> H[Similarity Matching]

H --> I[ATS Score]
H --> J[Resume Ranking]
H --> K[Missing Skills]
H --> L[AI Feedback]

I --> M[Streamlit Dashboard]
J --> M
K --> M
L --> M
```
