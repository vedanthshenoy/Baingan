# Baingan

*A collaborative prompt manager with active scoring*

---

## Overview

Baingan is an **LLM prompt management and experimentation tool**.  
It helps you **design, test, compare, and refine prompts** in a collaborative way.  

The main idea is:  
- You give Baingan some **prompts** and a **query/output** you expect.  
- Baingan lets you run prompts **individually, chained, or combined**.  
- You can **edit outputs**, and Baingan will:  
  - Suggest **better prompts** to get closer to your edited output.  
  - **Score** how likely existing prompts would have produced that output.  
- It provides a **slider** to assign importance/weight to different prompts.  
- You can save results and export them to analyze later.  

In short, **Baingan = A playground to experiment with prompts + a feedback loop for improving them**.  

---

## Features

| Feature | Status |
|---|---|
| Run prompts individually, chained, or combined | ✔️ |
| Slider to assign weight/importance to prompts | ✔️ |
| Edit output & get recommended prompt changes | ✔️ |
| Score (%) of how likely existing prompts produce edited output | X |
| Export results (dataframe/Excel) | ✔️ |
| Submit score to a response | ✔️ |
| Visualize which prompt influenced which part of output | ⚙️ Experimental |

---

## Architecture

- **Frontend**: Streamlit app (`advanced_baingan_app.py`)  
- **Backend**: Python logic for chaining, combining, scoring prompts  
- **LLM Integration**: Uses your own Chat/RAG endpoint (Baingan sends queries to your LLM and scores results)  
- **Storage**: In-memory dataframes (with export support)  

---

## Getting Started

### Prerequisites

- Python >= 3.9  
- `pip` package manager  
- An LLM / RAG endpoint (e.g. OpenAI, Gemini, custom FastAPI server)  
- API key / credentials if required  

### Installation

```bash
git clone https://github.com/vedanthshenoy/Baingan.git
cd Baingan
pip install -r requirements.txt
```

### Configuration

- Ensure your LLM / RAG endpoint is running  
- Provide API key if required (via `.env` or directly in app config)  

### Running the App

```bash
streamlit run project_files/advanced_baingan_app.py
```
Then open the Streamlit URL in your browser.  

---

## Usage

1. Enter your **query** and **system prompts**.  
2. Choose a **mode**:
   - **Individual**: Test each prompt separately  
   - **Chained**: Feed output of one prompt into the next  
   - **Combined**: Run multiple prompts together  
3. Optionally **edit outputs** → Baingan recommends better prompts.  
4. View **scores** to see how well prompts align with your target output.  
5. Save/export results for later use.  

---

## Mode Descriptions

- **Individual Mode** → Evaluate prompts in isolation.  
- **Chained Mode** → Multi-step workflows where one output feeds into another prompt.  
- **Combined Mode** → Multiple prompts are applied together in a single context.  

---

## Development

- Fork the repo  
- Create a feature branch  
- Submit PRs with changes  

### Ideas for contributions
- More visualization of prompt → output influence  
- Persistent storage (database) for sessions  
- Support for more LLM providers  
- UI/UX polish  

---

## License

This project is licensed under the **GPL-2.0** License.  

---

## Contact

**Author**: [Vedanth Shenoy](https://github.com/vedanthshenoy)  

For questions, open an **Issue** in the repo.  



Happy Prompt Engineering with Baingan 🍆🪄✨️
