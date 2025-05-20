"""Database connection and interaction logic using SQLAlchemy for async SQLite."""

from common import config
from common.llm_factory import get_llm
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import logging

logger = logging.getLogger(__name__)

# Singleton for SQLDatabase
_sql_database = None

def get_sql_database():
    """Initializes and returns the SQLDatabase instance (singleton)."""
    global _sql_database
    if _sql_database is None:
        if not config.FEYOD_DATABASE_URL:
            logger.error("FEYOD_DATABASE_URL is not configured. Cannot create SQLDatabase instance.")
            raise ValueError("Database URL is not configured.")
        try:
            logger.info(f"Creating SQLDatabase instance for URL: {config.FEYOD_DATABASE_URL}")
            _sql_database = SQLDatabase.from_uri(config.FEYOD_DATABASE_URL)
        except Exception as e:
            logger.exception(f"Failed to create SQLDatabase instance: {e}")
            raise
    return _sql_database

def get_toolkit():
    """Initializes and returns the SQLDatabaseToolkit instance."""
    sql_database = get_sql_database()
    llm = get_llm()
    return SQLDatabaseToolkit(db=sql_database, llm=llm)

def get_schema_description() -> str:
    """Retrieves a string description of the database schema using toolkit tools."""
    try:
        toolkit = get_toolkit()
        tables_tool = next(tool for tool in toolkit if tool.name == "sql_db_list_tables")
        schema_tool = next(tool for tool in toolkit if tool.name == "sql_db_schema")
        tables = tables_tool.run()
        schema = schema_tool.run(tables)
        logger.info("Successfully retrieved schema description.")
        return schema
    except Exception as e:
        logger.error(f"Error retrieving schema description: {e}")
        return f"Error retrieving schema description: {e}"

def validate_query(sql: str) -> str:
    """Validates a given SQL query using QuerySQLCheckerTool from the toolkit."""
    try:
        toolkit = get_toolkit()
        checker_tool = next(tool for tool in toolkit if tool.name == "sql_db_query_checker")
        validation_result = checker_tool.run(sql)
        if "Error" in validation_result:
            raise ValueError(f"Query validation failed: {validation_result}")
        logger.info("Query validation successful.")
        return validation_result
    except Exception as e:
        logger.error(f"Error validating query: {e}")
        raise

def execute_query(sql: str) -> str:
    """Executes a given SQL query using the toolkit's query tool and returns results as JSON."""
    try:
        toolkit = get_toolkit()
        query_tool = next(tool for tool in toolkit if tool.name == "sql_db_query")
        results = query_tool.run(sql)
        logger.info(f"Query executed successfully: {results}")
        return results
    except Exception as e:
        logger.error(f"Unexpected error executing query: {e}")
        raise ValueError(f"Unexpected error executing query: {e}")

# Example usage (for testing purposes) - Requires running in an async context
async def _test_main():
    logging.basicConfig(level=logging.INFO)
    # Set FEYOD_DATABASE_URL environment variable before running for this test
    if not config.FEYOD_DATABASE_URL:
        print("Please set the FEYOD_DATABASE_URL environment variable to run the test.")
        return

    print("Testing SQLDatabase connection...")
    try:
        get_sql_database()
        print("SQLDatabase instance created successfully.")
    except Exception as e:
        print(f"Connection failed: {e}")


    print("\nFetching schema...")
    try:
        schema = await get_schema_description()
        print(schema)
    except Exception as e:
        print(f"Schema fetch failed: {e}")


    print("\nTesting query execution (example: first 5 clubs)...")
    try:
        clubs = execute_query("SELECT clubId, clubName FROM clubs LIMIT 5;")
        print(clubs)
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    import asyncio
    print("Running database test function. Ensure FEYOD_DATABASE_URL is set in your environment.")
    asyncio.run(_test_main())