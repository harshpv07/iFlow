import streamlit as st
import os
import subprocess
from openai import OpenAI
from typing import Annotated, Sequence, TypedDict, Union
from langgraph.graph import Graph, StateGraph, START, END
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
    result = workflow.invoke({"query": data, "messages": [], "script": None, "execution_result": None})
    return jsonify(result)


# Load environment variables
load_dotenv()

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[ToolInvocation], "messages"]
    query: str
    script: Union[str, None]
    execution_result: Union[int, str, None]


def script_validator_agent(state: AgentState) -> AgentState:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )

    messages = [
        {
            "role": "user", 
            "content": f"""
                Based on the input query, determine one of the following from the list and only return the number of the option that best describes the query.
                List:
                1. Is this task related to opening a specific application or can this task be converted into Windows batch script (.cmd) commands? 
                2. Is this task related to emails search or reading the emails of the user? 
                3. Is this task related to notes search or reading the notes of the user?
                4. Is this task related to gallery search or searching for images of the user or reading the gallery of the user?
                5. Is this task related to checking the videos of the user or opening the camera of the user? 
                
            Go through all the options and return a single option number from the list. 
            Answer only the number of the option from the list without any further reasoning. 
            Query: {state['query']}

            """
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
    print("#################### This is the option number selected by the user ######################", response)
    return {"messages": state["messages"], "query": state["query"], "script": None, "execution_result": response}


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
    print("###################### This is the script content ######################", script_content)
    return {"messages": state["messages"], "query": state["query"], "script": script_content, "execution_result": None}


def input_collector_agent(state: AgentState) -> AgentState:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
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


def search_milvus_agent(state: AgentState) -> AgentState:
    """Agent to search Milvus vector store based on query and process results with LLM"""
    from pymilvus import connections, Collection
    import numpy as np
    from transformers import CLIPProcessor, CLIPModel
    from openai import OpenAI

    try:
        # Initialize CLIP model and processor for query embedding
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")

        # Get query embedding
        query = state["query"]
        inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)
        text_features = model.get_text_features(**inputs)
        query_embedding = text_features.detach().numpy()[0]

        # Search only notes collection
        collection_name = "clip_notes"
        collection = Collection(collection_name)
        collection.load()

        # Perform similarity search
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results_collection = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=5,
            output_fields=["text", "metadata"]
        )

        # Format results
        results = []
        for hits in results_collection:
            for hit in hits:
                results.append({
                    'text': hit.entity.get('text'),
                    'metadata': hit.entity.get('metadata'),
                    'score': hit.score,
                    'collection': 'notes'
                })

        # Sort results by score
        results.sort(key=lambda x: x['score'])

        # Process results with LLaMA
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )

        # Format results for LLM prompt
        context = "Here are the relevant documents found:\n\n"
        for r in results[:3]:  # Use top 3 results
            context += f"Document from {r['collection']}:\n{r['text']}\n\n"

        messages = [
            {
                "role": "user",
                "content": f"""Based on the following search results, please:
                1. Summarize the key information relevant to the query
                2. Suggest 2-3 follow-up questions the user might want to ask
                3. Highlight any important dates, names, or action items found

                Query: {query}

                {context}"""
            }
        ]

        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024
        )

        llm_response = completion.choices[0].message.content.strip()
        
        return {
            "messages": state["messages"], 
            "query": state["query"],
            "script": state["script"],
            "execution_result": {
                "raw_results": results,
                "llm_analysis": llm_response
            }
        }

    except Exception as e:
        return {
            "messages": state["messages"],
            "query": state["query"], 
            "script": state["script"],
            "execution_result": f"Search failed: {str(e)}"
        }


def search_milvus_agent_email(state: AgentState) -> AgentState:
    """Agent to search Milvus vector store specifically for emails"""
    from pymilvus import connections, Collection
    import numpy as np
    from transformers import CLIPProcessor, CLIPModel
    from openai import OpenAI

    try:
        # Initialize CLIP model and processor for query embedding
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")

        # Get query embedding
        query = state["query"]
        inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)
        text_features = model.get_text_features(**inputs)
        query_embedding = text_features.detach().numpy()[0]

        # Search only emails collection
        collection_name = "clip_emails"
        collection = Collection(collection_name)
        collection.load()

        # Perform similarity search
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=5,
            output_fields=["text", "metadata"]
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'text': hit.entity.get('text'),
                    'metadata': hit.entity.get('metadata'),
                    'score': hit.score,
                    'collection': 'emails'
                })
                    
        # Sort results by score
        formatted_results.sort(key=lambda x: x['score'])

        # Process results with LLaMA
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )

        # Format results for LLM prompt
        context = "Here are the relevant emails found:\n\n"
        for r in formatted_results[:3]:  # Use top 3 results
            context += f"Email {r['metadata'].get('subject', 'No subject')}:\n{r['text']}\n\n"

        messages = [
            {
                "role": "user",
                "content": f"""Based on the following email search results, please:
                1. Summarize the key information from the emails
                2. Extract any important dates, names, or action items
                3. Note any important follow-up actions needed

                Query: {query}

                {context}"""
            }
        ]

        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024
        )

        llm_response = completion.choices[0].message.content.strip()
        
        return {
            "messages": state["messages"], 
            "query": state["query"],
            "script": state["script"],
            "execution_result": {
                "raw_results": formatted_results,
                "llm_analysis": llm_response
            }
        }

    except Exception as e:
        return {
            "messages": state["messages"],
            "query": state["query"], 
            "script": state["script"],
            "execution_result": f"Email search failed: {str(e)}"
        }


def create_workflow() -> Graph:
    # Create workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("validate", script_validator_agent)
    workflow.add_node("input_collector", input_collector_agent)
    workflow.add_node("generate", script_generator_agent)
    workflow.add_node("execute", execution_agent)
    workflow.add_node("search", search_milvus_agent)
    workflow.add_node("search_emails", search_milvus_agent_email)

    # Set up edges
    workflow.add_edge("input_collector", "generate")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", END)
    workflow.add_edge("search", END)
    workflow.add_edge("search_emails", END)

    # Conditional routing
    def route_branch(state: AgentState) -> str:
        match state["execution_result"]:
            case "1":
                return "input_collector"
            case "2":
                return "search_emails"
            case "3":
                return "search"
            case _:
                return END

    workflow.add_conditional_edges(
        "validate",
        route_branch,
        {
            "input_collector": "input_collector",
            "search_emails": "search_emails",
            "search": "search",
            END: END
        }
    )

    workflow.set_entry_point("validate")
    return workflow.compile()

from flask import jsonify

if __name__ == "__main__":
    def test_workflow():
        workflow = create_workflow()
        result = workflow.invoke({"query": "Summarize all my emails", "messages": [], "script": None, "execution_result": None})
        print(result)
    test_workflow()
    #app.run(port=8501 , debug=True)