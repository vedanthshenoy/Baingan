# Baingan 🍆

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

## What Baingan 🍆 does

### Business Problem / Outcome

Modern AI / LLM (Large Language Model) users often need to craft *prompts* (the instructions or questions you feed into an AI) that get good, reliable, useful responses. But there are challenges:

- It’s hard to know **which prompt version** works best.  
- Sometimes, you want to combine prompts, chain them together (one prompt → its output feeds into another), or weight them (some prompt’s significance higher than others).  
- If you edit the AI’s response manually, you want to understand: _How far is that from what the AI would produce using existing prompts?_  
- You want feedback loops: when you change outputs, what prompt changes could lead to that better output?  

The business or user outcomes here are:

- Faster iteration over prompt design (less trial & error).  
- More predictable, higher quality AI outputs.  
- Better insights into how prompts contribute to results → smarter improvements.  
- Ability to compare prompt variants, combine them, or chain them, all in one tool.

---

### How Baingan 🍆 Solves It

Here’s how this app helps (step by step, in simple terms):

1. **User interface via Streamlit**  
   The code provides a UI where you can input your query, your prompts, choose modes, etc. Streamlit makes it easy to build interactive apps in Python without heavy front-end work.

2. **Multiple modes of prompt evaluation**  
   - **Individual Mode**: Run each prompt separately to see what each produces.  
   - **Chained Mode**: Feed output from one prompt into another to build multi-step logic.  
   - **Combined Mode**: Use prompts together (e.g. more than one prompt in a context) to see how their combined effect works.

   This helps you see not just one prompt’s effect, but how prompts interact.

3. **Editing output + feedback**  
   If the AI response isn’t quite right, you can manually edit it. Then the app looks at your edited version and gives:  
   - Scores — how likely each existing prompt might have produced something like your edited output.  
   - Suggestions — changes or improved prompts that might lead the AI to produce output closer to your edited version.

4. **Weighting / importance sliders**  
   You can assign weights to different prompts, so the system knows which you consider more important. This influences how combinations or comparisons are done.

5. **Data export and tracking**  
   You can save or export the results (like into spreadsheets) so you can compare across prompt designs over time. Useful for teams or for keeping history.

---

### Why This Helps Beginners & Teams

- You don’t need to guess blindly which prompt works best. You can test many variants and get feedback.  
- You can visually compare results, see how changing one prompt or chaining prompts changes the output.  
- If you're new to prompt engineering, this gives structure: you can try simple → combine → refine.  
- For teams, this enables shared evaluation: someone can edit outputs, someone else can see what prompt changes worked, etc.

---

## Summary

In short:

- **Problem**: Designing good prompts is hard, especially when you want reliable, predictable outputs and want to improve or refine them.

- **Solution**: This app gives you tools to run, compare, combine, chain, edit, score, and track prompts — all in one place.

- **Business Outcome**: More efficient prompt engineering, better AI outputs, faster iteration, and clearer insight into what’s working and what needs improvement.

---

## Example Flow (Beginner Version)

Here’s a simple example of how a new user might use this app:

1. They write two or three prompts for the same query (“summarize this article”, “summarize and highlight insights”, etc.).  
2. Run them in *Individual Mode* to see which summaries look better.  
3. Maybe combine them in *Combined Mode* or chain them: first prompt extracts insights, second builds summary from those insights.  
4. They don’t like one summary → they edit it manually. The app suggests how to change the prompts so future outputs are closer.  
5. They assign weights: maybe the insights-extraction prompt is more important to them.  
6. They export the results, share with teammates, iterate again.

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
