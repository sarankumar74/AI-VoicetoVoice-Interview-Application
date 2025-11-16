# Interview Analyzer: Real-Time Conversation Analyzer

![Streamlit](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Google Gemini API](https://img.shields.io/badge/google-gemini?logo=mysql&logoColor=white)
![API Handling](https://img.shields.io/badge/Process-API%20Handling-brightgreen)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![Domain](https://img.shields.io/badge/AI-Interview%2C%HR%2C%Buissnes-Purpose)

---

**Transforming Interviews and Discussions with AI-driven Feedback**  

Interviews and group discussions are crucial in hiring and learning, but human evaluation can be subjective, inconsistent, and slow. **Interview Analyzer** leverages AI to listen, understand, and provide **instant, structured feedback** on candidate interactions—just like having a real mentor at your side.  

---

## 🚀 Purpose

Build an AI Interview Analyzer that:  

- Observes and understands candidate interactions in real-time  
- Generates insightful, data-driven feedback  
- Helps users improve tone, clarity, confidence, and communication skills  

This tool aims to make interviews **fairer, faster, and more objective**, providing actionable insights for both one-on-one interviews and group discussions.  

---

## 🎯 Problem Statement

Design an **AI-driven Interview Analysis System** that:  

- Processes spoken or written conversations  
- Generates insightful feedback reports including:  
  - Tone, sentiment, and clarity of participants  
  - Evaluation of confidence, empathy, and communication quality  
  - Summarization of key discussion points  
  - Suggestions for improvement  

---

## ⚙️ Core Features

### 1. Input Processing
- Accepts either:  
  - Audio recordings of discussions/interviews  
  - Live webcam sessions  
- Allows domain and round selection (Tech, Managerial, HR, Group Discussion) for context-aware feedback  

### 2. Speech-to-Speak
- Converts audio to text using **Google Gemini API** for accurate transcription  

### 3. AI Analysis Layer
- **Sentiment & Tone Detection**: Positive, Neutral, Negative, Confident, Nervous, etc.  
- **Empathy & Emotional Intelligence Analysis**  
- **Clarity & Coherence Scoring**: Grammar, filler words, pace, etc.
- **Topic / Keyword Extraction**  

### 4. Summarization & Insights
- Generates:  
  - Summary of overall discussion  
  - Key points mentioned by each participant  
  - Suggested areas of improvement (e.g., “Be more concise”, “Structure answers better”)  

### 5. Output Format
- Detailed **reports** (text or dashboard) showing:  
  - Sentiment trends  
  - Confidence score per speaker  
  - Overall summary  
  - Suggested improvement actions  
- **Real-time dashboard visualizations**  
  - Sentiment over time  
  - Confidence vs. Time graphs  
- Auto-generated **Mock Interview Summary PDF**  
- Feedback Tone Customization: Professional, Encouraging, Critical  

---


<summary>📸 Click to view Streamlit UI screenshots</summary>

#### Home Page  
![Home Page](https://github.com/user-attachments/assets/12faa08f-51d7-422e-b595-307bdfafaedc)


#### AI Voice & Web access Interview  Page  
![Result Page](https://github.com/user-attachments/assets/fbe43007-2131-402c-a3af-ced5d74f985c)


####  Interwier Answer Page  1 
![Result Page](https://github.com/user-attachments/assets/d33d04b6-ce4e-4c42-be75-8ac3b5a3c80e)


####  Interwier Answer Page  1  
![Result Page](https://github.com/user-attachments/assets/2ce37fad-d21e-4561-a540-e795b0c8e218)


####  Score Page   
![Result Page](https://github.com/user-attachments/assets/2e9fc5cf-6360-44d0-817c-9dc085d5aa85)


#### Voice Transription Replay Page  
![Result Page](https://github.com/user-attachments/assets/afe0cbba-f246-42a3-b6c2-2aaf807b7849)


#### Charts Score  Page  
![Result Page](https://github.com/user-attachments/assets/7ac7efa6-63b7-404f-92d5-92cb1385aba9)


---

## 🧩 Project Structure
```bash

Interview-Analyzer/
│
├─ app.py # Main Streamlit app
│
├─ requirements.txt # Project dependencies
│
└─ README.md
