import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
from contextlib import AsyncExitStack

# Load .env variables
load_dotenv()

app = FastAPI()
loop = asyncio.get_event_loop()

class QueryRequest(BaseModel):
    query: str

class MCPFastAPIClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    async def start(self, server_script: str):
        if not server_script.endswith(".py"):
            raise ValueError("Invalid server script")

        server_params = StdioServerParameters(command="python", args=[server_script])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        print("MCP Server connected.")

    async def process_query(self, query: str) -> str:
        tool_response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in tool_response.tools]

        system_prompt = (
            "You are an intelligent assistant connected to an MCP tool server. "
            "When a user asks a question, first check if any of the available tools are suitable to help answer it. "
            "If a tool is useful, call it with the appropriate arguments. Then, based on the tool's result, "
            "generate a short and informative response in natural language for the user. "
            "Here are the available tools: " + ", ".join([t['function']['name'] for t in available_tools]) + "."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        while True:
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                    temperature=0.5,
                    max_completion_tokens=1024
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Groq API Error: {e}")

            message = response.choices[0].message
            print("LLM RESPONSE:", message)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except Exception as e:
                        print(f"Failed to parse arguments for {tool_name}: {e}")
                        continue

                    print(f"Calling tool: {tool_name} with args: {tool_args}")
                    try:
                        tool_result = await self.session.call_tool(tool_name, tool_args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result.content)
                        })
                    except Exception as e:
                        print(f"Tool call failed: {e}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {e}"
                        })

                # Loop again to get the LLM's follow-up response
                continue

            # No tool call, just return final content
            if message.content:
                return message.content.strip()
            else:
                return "No response generated."


    async def close(self):
        await self.exit_stack.aclose()

# Global instance
mcp_client = MCPFastAPIClient()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await mcp_client.start("ragServer.py")
    yield
    # Shutdown
    await mcp_client.close()

app = FastAPI(lifespan=lifespan)

@app.post("/query")
async def query_handler(req: QueryRequest):
    response = await mcp_client.process_query(req.query)
    return {"response": response}
