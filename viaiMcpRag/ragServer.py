import logging
import time
from mcp.server.fastmcp import FastMCP
from vectorstore_utils import VectorStoreBuilder

# Setup logging
log = logging.getLogger("MCP_RAG")
logging.basicConfig(level=logging.INFO)

# Initialize FastMCP and VectorStoreBuilder
mcp = FastMCP("rag")
builder = VectorStoreBuilder()

log.info("Loading vector store...")
start_time = time.time()

vectorstore = builder.get_vectorstore()

# Convert the vectorstore into a retriever for querying
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

log.info(f"Vector store ready in {time.time() - start_time:.2f} seconds.")

@mcp.tool()
async def retrieve_doc(query: str) -> str:
    log.info(f"Tool call: retrieve_doc(query='{query}')")
    start = time.time()

    # Retrieve relevant documents based on the query
    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return "No relevant documents found for your query."

    duration = time.time() - start
    log.info(f"Retrieved {len(relevant_docs)} documents in {duration:.2f} seconds.")

    # Format the retrieved documents into a readable string
    return "\n\n".join([
        f"==DOCUMENT {i+1}==\nSource: {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}"
        for i, doc in enumerate(relevant_docs)
    ])

if __name__ == "__main__":
    log.info("Starting RAG MCP server...")
    mcp.run(transport="stdio")
    log.info("Started RAG MCP server...")

