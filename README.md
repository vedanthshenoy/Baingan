# Baingan
A collaborative prompt manager with active scoring 

1. Give option and check chaining, combining and ordering( for individual runs as well as  as of multiple prompts in same context) prompts by teammates : ✔️
2. Have slider to add weightage to prompts : ✔️
3. Allow to edit output however needed and get prompt change recommendations ✔️
4. For the manually edited output, run the exisiting prompts in the dashboard and give scores in % how much that prompt will be able to create an output like this 
5. Experimental : Allow users to visualize which part of whose prompt influenced which parts of output


## How to Run It 🏃‍♂️

You can have your own Chat/RAG app with an endpoint exposed to get started. 
let me take you step by step
1. Install all requirements.txt
2. Run the streamlit app : advanced_baingan_app.py which is in the project_files folder
3. Add your Llm API key if not alresdy configured by the system. Usually we have our own Gemini API running for you.
4. Paralelly run your Chat/RAG app or have an endpoint. Paste that endpoint in the endpoint field in the Baingan app.
5. Fill in the details, in your query put what a user query might be.
6. Add your system prompts to test. you can assign it names as well.
7. Select the mode of testing : Individual, Chained, Combined.
8. Test the prompts and also edit and reverse the flow to get probable prompt for the given/edited output response.
9. Save it in a dataframe or export it as an excel.
10. Edit and keep testing your prompts until you obtain a satosfactory output.

Happy Prompt Engineering with Baingan 🍆🪄✨️
