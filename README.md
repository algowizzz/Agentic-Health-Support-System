# Intelligent Patient Risk Assessment & Agentic Health Support System

An AI-powered healthcare analytics system that predicts patient health risk using machine learning and extends into an intelligent agent-based assistant for structured health recommendations.

---

## Project Overview

Early detection of patient health risk is essential for preventive healthcare. Manual risk evaluation is time-consuming and often inconsistent.

This system automates risk assessment using traditional Machine Learning models and provides a scalable architecture for intelligent health guidance.

The project is divided into two milestones:

### Milestone 1 – ML-Based Risk Assessment (No GenAI)

Predicts patient health risk using classical ML algorithms.

### Milestone 2 – Agentic Health Assistant

Extends the system using an agent workflow to generate structured health reports and recommendations.

---

# Problem Statement

Healthcare institutions require efficient systems to:

- Identify high-risk patients early
- Provide interpretable risk scores
- Assist in preventive healthcare planning
- Maintain consistency in evaluation

This project builds a structured ML-based risk prediction system and extends it into an intelligent decision-support assistant.

---

# Key Features

## Milestone 1 – Machine Learning Risk Prediction

- Data cleaning & preprocessing
- Feature engineering
- Multiple ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Model comparison & evaluation
- Risk score generation (0–1 probability)
- Risk classification:
  - Low Risk
  - Medium Risk
  - High Risk
- Confusion matrix & performance metrics
- Feature importance visualization
- Interactive Streamlit interface
- Public deployment

---

## Milestone 2 – Agentic AI Extension

- Uses predicted risk output
- Generates structured health summary
- Provides preventive recommendations
- Suggests follow-up actions
- Medical disclaimer inclusion
- Optional Retrieval-Augmented Generation (RAG)

---

# System Architecture

## Milestone 1 Workflow

User Input  
↓  
Data Preprocessing  
↓  
Feature Engineering  
↓  
ML Model  
↓  
Risk Score + Category  
↓  
UI Display

---

## Milestone 2 Workflow

User Input  
↓  
ML Risk Prediction  
↓  
Agent Workflow  
↓  
Knowledge Retrieval  
↓  
Structured Health Report  
↓  
UI Display / Download

---

# Project Structure

# Installation & Setup

Follow these steps to set up the project on your local machine.

## Prerequisites

- **Python 3.8+** - Ensure you have Python 3.8 or higher installed
- **pip** - Package installer for Python

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Agentic-Health-Support-System.git
cd Agentic-Health-Support-System
```

## Step 2: Create a Virtual Environment

It is recommended to create a virtual environment to avoid dependency conflicts.

### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## Step 3: Install Dependencies

Install all required packages using the requirements.txt file:

```bash
pip install -r src/requirements.txt
```

## Step 4: Run the Application

Start the Streamlit application:

```bash
cd ui
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Additional Configuration

### Environment Variables

If needed, create a `.env` file in the project root with the following variables:

```
# Add any required environment variables here
```

### Data Files

Ensure the following directories exist with required data files:

- `data/` - Contains training and test datasets

## Troubleshooting

- **Module not found errors**: Make sure you've activated the virtual environment and installed all dependencies
- **Port already in use**: Use a different port with `streamlit run streamlit_app.py --server.port 8502`

---


