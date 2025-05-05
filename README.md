# feyod-chatbot-web

This repository contains the code for "Fred", a chatbot designed to answer questions about the Feyenoord football club using data from the [Feyod dataset](https://github.com/jeroenvdmeer/feyod). It leverages Chainlit for the user interface and LangGraph to manage the workflow of converting natural language questions into SQL queries, executing them, and formulating answers.

## Features

*   **Natural Language Understanding:** Interprets user questions about Feyenoord matches, players, statistics, etc.
*   **Database Interaction:** Connects to a SQLite database containing historical Feyenoord data. See the [`feyod` repository](https://github.com/jeroenvdmeer/feyod) for more information about the dataset. 
*   **NL-to-SQL Conversion:** Uses a Language Model (LLM) to translate natural language questions into SQL queries.
*   **Query Execution & Validation:** Executes the generated SQL queries against the database and includes basic syntax validation and fixing attempts.
*   **Answer Generation:** Formulates user-friendly answers based on the retrieved database results using an LLM.
*   **Web Interface:** Provides an interactive chat interface using Chainlit.

## Setup

Follow these steps to set up and run the chatbot locally:

1.  **Prerequisites:**
    *   Python 3.9+
    *   Git
    *   Access to an LLM provider (like OpenAI or Google) and an API key.

2.  **Clone the Repository:**
    ```bash
    # If you haven't cloned this repository yet
    git clone https://github.com/jeroenvdmeer/feyod-chatbot-web
    cd feyod-chatbot-web
    ```

3.  **Set up the Database:**
    *   This chatbot relies on the `feyod.db` database from the main `feyod` project. Follow the database setup instructions in the `feyod-common` README: [`feyod-common/README.md`](https://github.com/jeroenvdmeer/feyod-common#database-setup).
    *   Ensure the `feyod.db` file is accessible to this project.

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    *   Create a `.env` file in the root directory
    *   Add the necessary environment variables. Refer to the `feyod-common` README ([`feyod-common/README.md`](https://github.com/jeroenvdmeer/feyod-common#configuration)) for a complete list. Key variables include:
        ```dotenv
        # Example .env content:
        FEYOD_DATABASE_URL="sqlite+aiosqlite:///path/to/your/feyod/feyod.db" # Adjust the path!
        LLM_PROVIDER="openai" # Or "google", etc.
        LLM_API_KEY="your_llm_api_key"
        LLM_MODEL="o4-mini" # Or another compatible model
        # Add other variables as needed (e.g., for specific example sources)
        ```
    *   **Important:** Replace placeholders like `/path/to/your/feyod/feyod.db` and `your_llm_api_key` with your actual values. The path to `feyod.db` should be relative or absolute based on where you placed the database file during the setup.

## Running the Chatbot

Once the setup is complete, you can run the chatbot using Chainlit:

```bash
chainlit run app.py -w
```

The `-w` flag enables auto-reloading when code changes are detected. Open your web browser to the URL provided by Chainlit (usually `http://localhost:8000`).

## Configuration

*   **Chatbot Behavior:** Environment variables in the `.env` file control database connections, LLM selection, and other core functionalities (see `common/config.py`).
*   **Chainlit UI:** The Chainlit user interface can be customized via the `.chainlit/config.toml` file. You can change the chatbot's name ("Fred"), theme, add links, etc. See the [Chainlit documentation](https://docs.chainlit.io/config/toml) for details.
*   **Welcome Message:** The initial message displayed to the user can be modified by editing the `chainlit.md` file.

## Dependencies

*   **[`feyod-common`](https://github.com/jeroenvdmeer/feyod-common):** This project relies heavily on the shared components provided by the `feyod-common` package located in the `common/` directory. This package handles database interactions, LLM integration, configuration loading, and query processing logic.
*   **Chainlit:** Used for building the interactive web UI.
*   **LangGraph:** Used to define and manage the agent's workflow.
*   **LangChain:** Provides core abstractions for working with LLMs.
*   **SQLAlchemy:** Used for asynchronous database interactions.