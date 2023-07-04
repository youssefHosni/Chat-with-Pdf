# Chat-With-Pdf
This project highlights how to leverage a ChromaDB vector store in a Langchain pipeline to create a chat with a Pdf application. You can load in a pdf based document and use it alongside an LLM without fine-tuning. 

![alt_text](https://github.com/youssefHosni/Chat-with-Pdf/blob/main/PDF-Chat%20App.png)

# Startup 🚀
1. Create a virtual environment `python -m venv langchainenv`
2. Activate it: 
   - Windows:`.\langchainenv\Scripts\activate`
   - Mac: `source langchain/bin/activate'
3. Clone this repo `git clone https://github.com/nicknochnack/LangchainDocuments`
4. Go into the directory `cd LangchainDocuments`
5. Install the required dependencies `pip install -r requirements.txt`
6. Add your OpenAI APIKey to line 52 of `app.py`
7. Start the app `streamlit run app.py`
8. Load the Pdf you would like to ask questions
9. Ask questions and get the answers

   



