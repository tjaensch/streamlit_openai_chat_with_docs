# Streamlit OpenAI Chat With Documents Application

### Live Link
https://openai-chat-with-docs.streamlit.app/ 

### How it works
The application is using [OpenAI's gpt-4o-mini model](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) to answer questions about the content of one or more files that contain text. Uploaded files are chunked into smaller pieces and each piece is embedded using the [LangChain](https://python.langchain.com/docs/get_started/introduction.html) OpenAIEmbeddings() class. The embeddings are temporarily saved in a [Chroma](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma) vector store.
The user can then ask questions about the content of the data and the application will return an answer. You will need a valid [OpenAI API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) to use this application. Using the OpenAI API is not free and you will be charged for the number of tokens used. The application will show you the number of tokens used and the approximate cost of the embeddings as of September 2024.

### Gotchas
This is an experimental proof of concept (POC) application and [the results are not guaranteed to be accurate and could be misleading or even wrong](https://becominghuman.ai/why-large-language-models-like-chatgpt-are-bullshit-artists-c4d5bb850852). You can and should never blindly trust the answers given by [Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model). Always verify the answers yourself.

![Screenshot](screenshot.PNG)
