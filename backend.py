import streamlit as st
import os
import subprocess
from openai import OpenAI
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import Graph, StateGraph , START, END
from langgraph.prebuilt import ToolInvocation
from langgraph.checkpoint.memory import MemorySaver
# Define state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[ToolInvocation], "messages"]
    query: str
    script: str | None
    execution_result: bool | None

def main():
    st.title("Task Automation Assistant")
    
    # Create text input
    user_input = st.text_input("Enter your task:", "")
    
    # Create button
    if st.button("Execute Task"):
        if user_input:
            workflow = create_workflow()
            result = workflow.invoke({"query": user_input, "messages": [], "script": None, "execution_result": None})
            
            if result["execution_result"]:
                st.success("Task executed successfully!")
            else:
                st.warning("Try another query")
        else:
            st.warning("Please enter a task description!")


def script_validator_agent(state: AgentState) -> AgentState:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
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
            st.write("Command Output:")
            st.code(result.stdout)
            return {"messages": state["messages"], "query": state["query"], "script": script_content, "execution_result": True}
        else:
            return {"messages": state["messages"], "query": state["query"], "script": script_content, "execution_result": False}
            
    except Exception as e:
        st.error(f"Execution failed: {str(e)}")
        return {"messages": state["messages"], "query": state["query"], "script": script_content, "execution_result": False}

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

if __name__ == "__main__":
    main()