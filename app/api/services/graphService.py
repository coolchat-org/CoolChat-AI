from typing import Annotated, Any, Dict, TypedDict
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from app.api.services.chatService import (
    response_from_LLM,
    call_backend_transaction
)

# Define state schema
class AgentState(TypedDict):
    messages: list[HumanMessage | AIMessage]
    current_message: str
    should_create_transaction: bool
    transaction_data: Dict[str, Any] | None

# Tools definition
@tool("create_transaction")
def create_transaction(transaction_data: dict) -> str:
    """Create a backend transaction with the provided data"""
    try:
        result = call_backend_transaction(transaction_data)
        return f"Transaction created successfully: {result}"
    except Exception as e:
        return f"Failed to create transaction: {str(e)}"

def create_CHATBOT_graph() -> Graph:
    # Initialize workflow graph
    workflow = StateGraph(AgentState)
    
    # Define tools
    tools = [create_transaction]
    tool_executor = ToolExecutor(tools)

    # Define nodes
    async def generate_response(state: AgentState) -> AgentState:
        """Generate response using RAG"""
        response = await response_from_LLM(
            chat_id=state["chat_id"],
            query=state["current_message"],
            db_host=state["db_host"]
        )
        state["messages"].append(AIMessage(content=response))
        
        # Check if response indicates need for transaction
        if "create transaction" in response.lower():
            state["should_create_transaction"] = True
            # Extract transaction data from response
            # This is a simplified example - you'd need more sophisticated parsing
            state["transaction_data"] = {
                "type": "transaction",
                "amount": 100,  # Example value
                "description": response
            }
        
        return state

    async def handle_transaction(state: AgentState) -> AgentState:
        """Handle transaction creation if needed"""
        if state["transaction_data"]:
            result = create_transaction(state["transaction_data"])
            state["messages"].append(AIMessage(content=result))
        return state

    # Add nodes to graph
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("handle_transaction", handle_transaction)

    # Define edges
    workflow.set_entry_point("generate_response")
    workflow.add_edge("generate_response", "handle_transaction")
    workflow.add_edge("handle_transaction", "end")

    # Compile graph
    return workflow.compile()

# async def process_chat(chat_id: str, message: str, db_host: str):
#     graph = create_CHATBOT_graph()
    
#     initial_state = {
#         "messages": [],
#         "current_message": message,
#         "should_create_transaction": False,
#         "transaction_data": None,
#         "chat_id": chat_id,
#         "db_host": db_host
#     }
    
#     final_state = await graph.ainvoke(initial_state)
#     return final_state["messages"][-1].content