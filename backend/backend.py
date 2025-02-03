import streamlit as st
import os
import subprocess
from openai import OpenAI
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import Graph, StateGraph , START, END
from langgraph.prebuilt import ToolInvocation
from langgraph.checkpoint.memory import MemorySaver
# Load environment variables from .env
from dotenv import load_dotenv
from flask import Flask, request

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    data = request.json  # Get the JSON payload
    print(data)
    workflow = create_workflow()
    #result = workflow.invoke({"query": data["text"], "messages": [], "script": None, "execution_result": None})
    #return jsonify(result)
    return {"message": f"Received data: {data}"}


# Load environment variables
load_dotenv()

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[ToolInvocation], "messages"]
    query: str
    script: str | None
    execution_result: bool | None

def main():
    st.title("Task Automation Assistant")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "output" in message:
                st.code(message["output"])
    
    # Chat input
    user_input = st.chat_input("Enter your task:")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process the task
        workflow = create_workflow()
        result = workflow.invoke({"query": user_input, "messages": [], "script": None, "execution_result": None})
        
        # Display assistant response
        with st.chat_message("assistant"):
            if result["execution_result"]:
                st.write("Task executed successfully!")
                if result["script"]:
                    st.code(result["script"], language="batch")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Task executed successfully!",
                    "output": result["script"]
                })
            else:
                st.write("Executed Task")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Executed Task"
                })

def script_validator_agent(state: AgentState) -> AgentState:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key = "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"
    )

    messages = [
        {
            "role": "user", 
            "content": f"Can this task be converted into Windows batch script (.cmd) commands? Answer only yes/no: {state['query']}"
        }
    ]

    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024
    )

    response = completion.choices[0].message.content.strip().lower()
    
    if response == "yes":
        return {"messages": state["messages"], "query": state["query"], "script": None, "execution_result": None}
    else:
        return {"messages": state["messages"], "query": state["query"], "script": None, "execution_result": False}

def script_generator_agent(state: AgentState) -> AgentState:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )

    messages = [
        {
            "role": "user",
            "content": f"""Convert the following task into a Windows batch script (.cmd) commands. 
            Only respond with the actual commands, no explanations.
            Task: {state['query']}"""
        }
    ]

    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024
    )

    script_content = completion.choices[0].message.content.strip()
    print(script_content)
    return {"messages": state["messages"], "query": state["query"], "script": script_content, "execution_result": None}




def input_collector_agent(state: AgentState) -> AgentState:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key= "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"
    )

    messages = [
        {
            "role": "user",
            "content": f"""For the following task, determine if any additional user inputs are needed.
            If inputs are needed, list them one per line starting with '?'. If no inputs needed, respond with 'None'.
            Task: {state['query']}"""
        }
    ]

    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024
    )

    required_inputs = completion.choices[0].message.content.strip()
    
    if required_inputs.lower() == 'none':
        return {"messages": state["messages"], "query": state["query"], "script": None, "execution_result": None}
    
    # Collect additional inputs from user
    collected_inputs = {}
    for input_line in required_inputs.split('\n'):
        if input_line.startswith('?'):
            input_prompt = input_line[1:].strip()
            user_response = st.text_input(input_prompt)
            if user_response:
                collected_inputs[input_prompt] = user_response
    
    # Append collected inputs to original query
    enhanced_query = state["query"] + "\nAdditional inputs:\n"
    for prompt, value in collected_inputs.items():
        enhanced_query += f"{prompt}: {value}\n"
    
    return {"messages": state["messages"], "query": enhanced_query, "script": None, "execution_result": None}



def execution_agent(state: AgentState) -> AgentState:
    script_content = state["script"]
    try:
        # Create a temporary .cmd file
        temp_script_path = "temp_script.cmd"
        with open(temp_script_path, "w") as f:
            f.write(script_content)

        # Execute the script using subprocess
        result = subprocess.run(
            [temp_script_path], 
            capture_output=True,
            text=True,
            shell=True
        )

        # Clean up
        os.remove(temp_script_path)

        if result.returncode == 0:
            return {"messages": state["messages"], "query": state["query"], "script": script_content, "execution_result": True}
        else:
            return {"messages": state["messages"], "query": state["query"], "script": script_content, "execution_result": False}
            
    except Exception as e:
        st.error(f"Execution failed: {str(e)}")
        return {"messages": state["messages"], "query": state["query"], "script": script_content, "execution_result": False}


def check_onenote(state : AgentState) -> AgentState:
    # Initialize Pinecone client
    import pinecone
    client = OpenAI()
    
    # Initialize Pinecone with API key from environment variable
    pinecone.init(api_key=os.getenv("pcsk_4b5HQc_A2QXPPwWyoqVCsXMEhQdWuXVn9SCfbyx2CRv4CZAQx86wzPtDdhfuP7J2KxUtDP"))
    
    # Create embedding for the query
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=state["query"]
    ).data[0].embedding

    # Query Pinecone index
    index = pinecone.Index("iflow")
    results = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True
    )

    # Extract relevant context from results
    context = ""
    for match in results.matches:
        if match.score > 0.7:  # Only include high relevance matches
            context += match.metadata.get("text", "") + "\n"

    # Return state with retrieved context
    return {
        "messages": state["messages"],
        "query": state["query"],
        "script": state["script"],
        "execution_result": state["execution_result"],
        "context": context
    }

    

def create_workflow() -> Graph:
    # Create workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("validate", script_validator_agent)
    workflow.add_node("generate", script_generator_agent)  
    workflow.add_node("execute", execution_agent)
    workflow.add_node("end", lambda x: x)

    # Add edges
    workflow.add_edge("validate", "generate")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", END)
    
    # Set entry point
    workflow.add_edge(START, "validate")
    
    return workflow.compile()

from flask import jsonify

if __name__ == "__main__":
    app.run(port=8501 , debug=True)