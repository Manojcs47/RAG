# Hugging Face Agents Course — Unit 3: Use Case for Agentic RAG

Source: https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction


---

<!-- introduction -->

# Introduction to Use Case for Agentic RAG

![Agentic RAG banner](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit3/agentic-rag/thumbnail.jpg)

In this unit, we will help Alfred, our friendly agent who is hosting the gala, by using Agentic RAG to create a tool that can be used to answer questions about the guests at the gala. 

> [!TIP]
> This is a 'real-world' use case for Agentic RAG, that you could use in your own projects or workplaces. If you want to get more out of this project, why not try it out on your own use case and share in Discord?


You can choose any of the frameworks discussed in the course for this use case. We provide code samples for each in separate tabs.

## A Gala to Remember

Now, it's time to get our hands dirty with an actual use case. Let's set the stage!

**You decided to host the most extravagant and opulent party of the century.** This means lavish feasts, enchanting dancers, renowned DJs, exquisite drinks, a breathtaking fireworks display, and much more.

Alfred, your friendly neighbourhood agent, is getting ready to watch over all of your needs for this party, and **Alfred is going to manage everything himself**. To do so, he needs to have access to all of the information about the party, including the menu, the guests, the schedule, weather forecasts, and much more!

Not only that, but he also needs to make sure that the party is going to be a success, so **he needs to be able to answer any questions about the party during the party**, whilst handling unexpected situations that may arise.

He can't do this alone, so we need to make sure that Alfred has access to all of the information and tools he needs.

First, let's give him a list of hard requirements for the gala.

## The Gala Requirements

A properly educated person in the age of the **Renaissance** needs to have three main traits.
He or she needed to be profound in the **knowledge of sports, culture, and science**. So, we need to make sure we can impress our guests with our knowledge and provide them with a truly unforgettable gala.
However, to avoid any conflicts, there are some **topics, like politics and religion, that are to be avoided at a gala.** It needs to be a fun party without conflicts related to beliefs and ideals.

According to etiquette, **a good host should be aware of guests' backgrounds**, including their interests and endeavours. A good host also gossips and shares stories about the guests with one another.

Lastly, we need to make sure that we've got **some general knowledge about the weather** to ensure we can continuously find a real-time update to ensure perfect timing to launch the fireworks and end the gala with a bang! 🎆

As you can see, Alfred needs a lot of information to host the gala.
Luckily, we can help and prepare Alfred by giving him some **Retrieval Augmented Generation (RAG) training!**

Let's start by creating the tools that Alfred needs to be able to host the gala!

---

<!-- agentic-rag -->

# Agentic Retrieval Augmented Generation (RAG)

In this unit, we'll be taking a look at how we can use Agentic RAG to help Alfred prepare for the amazing gala.

> [!TIP]
> We know we've already discussed Retrieval Augmented Generation (RAG) and agentic RAG in the previous unit, so feel free to skip ahead if you're already familiar with the concepts.

LLMs are trained on enormous bodies of data to learn general knowledge.
However, the world knowledge model of LLMs may not always be relevant and up-to-date information.
**RAG solves this problem by finding and retrieving relevant information from your data and forwarding that to the LLM.**

![RAG](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/rag.png)

Now, think about how Alfred works:

1. We've asked Alfred to help plan a gala
2. Alfred needs to find the latest news and weather information
3. Alfred needs to structure and search the guest information

Just as Alfred needs to search through your household information to be helpful, any agent needs a way to find and understand relevant data.
**Agentic RAG is a powerful way to use agents to answer questions about your data.** We can pass various tools to Alfred to help him answer questions.
However, instead of answering the question on top of documents automatically, Alfred can decide to use any other tool or flow to answer the question.

![Agentic RAG](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/agentic-rag.png)

Let's start **building our agentic RAG workflow!**  

First, we'll create a RAG tool to retrieve up-to-date details about the invitees. Next, we'll develop tools for web search, weather updates, and Hugging Face Hub model download statistics. Finally, we'll integrate everything to bring our agentic RAG agent to life!  

---

<!-- invitees -->

# Creating a RAG Tool for Guest Stories


Alfred, your trusted agent, is preparing for the most extravagant gala of the century. To ensure the event runs smoothly, Alfred needs quick access to up-to-date information about each guest. Let's help Alfred by creating a custom Retrieval-Augmented Generation (RAG) tool, powered by our custom dataset.

## Why RAG for a Gala?

Imagine Alfred mingling among the guests, needing to recall specific details about each person at a moment's notice. A traditional LLM might struggle with this task because:

1. The guest list is specific to your event and not in the model's training data
2. Guest information may change or be updated frequently
3. Alfred needs to retrieve precise details like email addresses

This is where Retrieval Augmented Generation (RAG) shines! By combining a retrieval system with an LLM, Alfred can access accurate, up-to-date information about your guests on demand.

> [!TIP]
> You can choose any of the frameworks covered in the course for this use case. Select your preferred option from the code tabs.

## Setting up our application

In this unit, we'll develop our agent within a HF Space, as a structured Python project. This approach helps us maintain clean, modular code by organizing different functionalities into separate files.  Also, this makes for a more realistic use case where you would deploy the application for public use.

### Project Structure

- **`tools.py`** – Provides auxiliary tools for the agent.  
- **`retriever.py`** – Implements retrieval functions to support knowledge access.  
- **`app.py`** – Integrates all components into a fully functional agent, which we'll finalize in the last part of this unit.  

For a hands-on reference, check out [this HF Space](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG), where the Agentic RAG developed in this unit is live. Feel free to clone it and experiment!

You can directly test the agent below:

<iframe
	src="https://agents-course-unit-3-agentic-rag.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

## Dataset Overview

Our dataset [`agents-course/unit3-invitees`](https://huggingface.co/datasets/agents-course/unit3-invitees/) contains the following fields for each guest:

- **Name**: Guest's full name
- **Relation**: How the guest is related to the host
- **Description**: A brief biography or interesting facts about the guest
- **Email Address**: Contact information for sending invitations or follow-ups

Below is a preview of the dataset:
<iframe
  src="https://huggingface.co/datasets/agents-course/unit3-invitees/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

> [!TIP]
> In a real-world scenario, this dataset could be expanded to include dietary preferences, gift interests, conversation topics to avoid, and other helpful details for a host.

## Building the Guestbook Tool

We'll create a custom tool that Alfred can use to quickly retrieve guest information during the gala. Let's break this down into three manageable steps:

1. Load and prepare the dataset
2. Create the Retriever Tool
3. Integrate the Tool with Alfred

Let's start with loading and preparing the dataset!

### Step 1: Load and Prepare the Dataset

First, we need to transform our raw guest data into a format that's optimized for retrieval.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

We will use the Hugging Face `datasets` library to load the dataset and convert it into a list of `Document` objects from the `langchain.docstore.document` module.

```python
import datasets
from langchain_core.documents import Document

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join(
            [
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}",
            ]
        ),
        metadata={"name": guest["name"]},
    )
    for guest in guest_dataset
]
```

</hfoption>
<hfoption id="llama-index">

We will use the Hugging Face `datasets` library to load the dataset and convert it into a list of `Document` objects from the `llama_index.core.schema` module.

```python
import datasets
from llama_index.core.schema import Document

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        text="\n".join(
            [
                f"Name: {guest_dataset['name'][i]}",
                f"Relation: {guest_dataset['relation'][i]}",
                f"Description: {guest_dataset['description'][i]}",
                f"Email: {guest_dataset['email'][i]}",
            ]
        ),
        metadata={"name": guest_dataset["name"][i]},
    )
    for i in range(len(guest_dataset))
]
```

</hfoption>
<hfoption id="langgraph">

We will use the Hugging Face `datasets` library to load the dataset and convert it into a list of `Document` objects from the `langchain.docstore.document` module.

```python
import datasets
from langchain_core.documents import Document

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join(
            [
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}",
            ]
        ),
        metadata={"name": guest["name"]},
    )
    for guest in guest_dataset
]
```

</hfoption>
</hfoptions>

In the code above, we:
- Load the dataset
- Convert each guest entry into a `Document` object with formatted content
- Store the `Document` objects in a list

This means we've got all of our data nicely available so we can get started with configuring our retrieval.

### Step 2: Create the Retriever Tool

Now, let's create a custom tool that Alfred can use to search through our guest information.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

We will use the `BM25Retriever` from the `langchain_community.retrievers` module to create a retriever tool.

> [!TIP]
> The <code>BM25Retriever</code> is a great starting point for retrieval, but for more advanced semantic search, you might consider using embedding-based retrievers like those from <a href="https://www.sbert.net/">sentence-transformers</a>.

```python
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = (
        "Retrieves detailed information about gala guests based on their name or relation."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."


# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)
```

Let's understand this tool step-by-step: 
- The `name` and `description` help the agent understand when and how to use this tool
- The `inputs` define what parameters the tool expects (in this case, a search query)
- We're using a `BM25Retriever`, which is a powerful text retrieval algorithm that doesn't require embeddings
- The `forward` method processes the query and returns the most relevant guest information

</hfoption>
<hfoption id="llama-index">

We will use the `BM25Retriever` from the `llama_index.retrievers.bm25` module to create a retriever tool.

> [!TIP]
> The <code>BM25Retriever</code> is a great starting point for retrieval, but for more advanced semantic search, you might consider using embedding-based retrievers like those from <a href="https://www.sbert.net/">sentence-transformers</a>.

```python
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever

bm25_retriever = BM25Retriever.from_defaults(nodes=docs)


def get_guest_info_retriever(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.retrieve(query)
    if results:
        return "\n\n".join([doc.text for doc in results[:3]])
    else:
        return "No matching guest information found."


# Initialize the tool
guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever)
```

Let's understand this tool step-by-step. 
- The docstring helps the agent understand when and how to use this tool
- The type decorators define what parameters the tool expects (in this case, a search query)
- We're using a `BM25Retriever`, which is a powerful text retrieval algorithm that doesn't require embeddings
- The method processes the query and returns the most relevant guest information

</hfoption>
<hfoption id="langgraph">

We will use the `BM25Retriever` from the `langchain_community.retrievers` module to create a retriever tool.

> [!TIP]
> The <code>BM25Retriever</code> is a great starting point for retrieval, but for more advanced semantic search, you might consider using embedding-based retrievers like those from <a href="https://www.sbert.net/">sentence-transformers</a>.

```python
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool

bm25_retriever = BM25Retriever.from_documents(docs)


def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."


guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation.",
)
```

Let's understand this tool step-by-step. 
- The `name` and `description` help the agent understand when and how to use this tool
- The type decorators define what parameters the tool expects (in this case, a search query)
- We're using a `BM25Retriever`, which is a powerful text retrieval algorithm that doesn't require embeddings
- The method processes the query and returns the most relevant guest information


</hfoption>
</hfoptions>

### Step 3: Integrate the Tool with Alfred

Finally, let's bring everything together by creating our agent and equipping it with our custom tool:

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import CodeAgent, InferenceClientModel

# Initialize the Hugging Face model
model = InferenceClientModel()

# Create Alfred, our gala agent, with the guest info tool
alfred = CodeAgent(tools=[guest_info_tool], model=model)

# Example query Alfred might receive during the gala
response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("🎩 Alfred's Response:")
print(response)
```

Expected output:

```
🎩 Alfred's Response:
Based on the information I retrieved, Lady Ada Lovelace is an esteemed mathematician and friend. She is renowned for her pioneering work in mathematics and computing, often celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine. Her email address is ada.lovelace@example.com.
```

What's happening in this final step:
- We initialize a Hugging Face model using the `InferenceClientModel` class
- We create our agent (Alfred) as a `CodeAgent`, which can execute Python code to solve problems
- We ask Alfred to retrieve information about a guest named "Lady Ada Lovelace"

</hfoption>
<hfoption id="llama-index">

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create Alfred, our gala agent, with the guest info tool
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm,
)

# Example query Alfred might receive during the gala
response = await alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("🎩 Alfred's Response:")
print(response)
```

Expected output:

```
🎩 Alfred's Response:
Lady Ada Lovelace is an esteemed mathematician and friend, renowned for her pioneering work in mathematics and computing. She is celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine. Her email is ada.lovelace@example.com.
```

What's happening in this final step:
- We initialize a Hugging Face model using the `HuggingFaceInferenceAPI` class
- We create our agent (Alfred) as a `AgentWorkflow`, including the tool we just created
- We ask Alfred to retrieve information about a guest named "Lady Ada Lovelace"

</hfoption>
<hfoption id="langgraph">

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool]
chat_with_tools = chat.bind_tools(tools)


# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }


## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
```

Expected output:

```
🎩 Alfred's Response:
Lady Ada Lovelace is an esteemed mathematician and pioneer in computing, often celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine.
```

What's happening in this final step:
- We initialize a Hugging Face model using the `HuggingFaceEndpoint` class. We also generate a chat interface and append the tools.
- We create our agent (Alfred) as a `StateGraph`, that combines 2 nodes (`assistant`, `tools`) using an edge
- We ask Alfred to retrieve information about a guest named "Lady Ada Lovelace"

</hfoption>
</hfoptions>

## Example Interaction

During the gala, a conversation might flow like this:

**You:** "Alfred, who is that gentleman talking to the ambassador?"

**Alfred:** *quickly searches the guest database* "That's Dr. Nikola Tesla, sir. He's an old friend from your university days. He's recently patented a new wireless energy transmission system and would be delighted to discuss it with you. Just remember he's passionate about pigeons, so that might make for good small talk."

```json
{
    "name": "Dr. Nikola Tesla",
    "relation": "old friend from university days",  
    "description": "Dr. Nikola Tesla is an old friend from your university days. He's recently patented a new wireless energy transmission system and would be delighted to discuss it with you. Just remember he's passionate about pigeons, so that might make for good small talk.",
    "email": "nikola.tesla@gmail.com"
}
```

## Taking It Further

Now that Alfred can retrieve guest information, consider how you might enhance this system:

1. **Improve the retriever** to use a more sophisticated algorithm like [sentence-transformers](https://www.sbert.net/)
2. **Implement a conversation memory** so Alfred remembers previous interactions
3. **Combine with web search** to get the latest information on unfamiliar guests
4. **Integrate multiple indexes** to get more complete information from verified sources

Now Alfred is fully equipped to handle guest inquiries effortlessly, ensuring your gala is remembered as the most sophisticated and delightful event of the century!

> [!TIP]
> Try extending the retriever tool to also return conversation starters based on each guest's interests or background. How would you modify the tool to accomplish this?
>
> When you're done, implement your guest retriever tool in the <code>retriever.py</code> file.

---

<!-- agent -->

# Creating Your Gala Agent

Now that we've built all the necessary components for Alfred, it's time to bring everything together into a complete agent that can help host our extravagant gala. 

In this section, we'll combine the guest information retrieval, web search, weather information, and Hub stats tools into a single powerful agent.

## Assembling Alfred: The Complete Agent

Instead of reimplementing all the tools we've created in previous sections, we'll import them from their respective modules which we saved in the `tools.py` and `retriever.py` files.

> [!TIP]
> If you haven't implemented the tools yet, go back to the <a href="./tools">tools</a> and <a href="./invitees">retriever</a> sections to implement them, and add them to the <code>tools.py</code> and <code>retriever.py</code> files.

Let's import the necessary libraries and tools from the previous sections:

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
# Import necessary libraries
import random
from smolagents import CodeAgent, InferenceClientModel

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset
```

Now, let's combine all these tools into a single agent:

```python
# Initialize the Hugging Face model
model = InferenceClientModel()

# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3,  # Enable planning every 3 steps
)
```

</hfoption>
<hfoption id="llama-index">

```python
# Import necessary libraries
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool
```

Now, let's combine all these tools into a single agent:

```python
# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create Alfred with all the tools
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)
```

</hfoption>
<hfoption id="langgraph">

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool
```

Now, let’s combine all these tools into a single agent:

```python
# Initialize the web search tool
search_tool = DuckDuckGoSearchRun()

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)


# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }


## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()
```
</hfoption>
</hfoptions>

Your agent is now ready to use!

## Using Alfred: End-to-End Examples

Now that Alfred is fully equipped with all the necessary tools, let's see how he can help with various tasks during the gala.

### Example 1: Finding Guest Information

Let's see how Alfred can help us with our guest information.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
query = "Tell me about 'Lady Ada Lovelace'"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```

Expected output:

```
🎩 Alfred's Response:
Based on the information I retrieved, Lady Ada Lovelace is an esteemed mathematician and friend. She is renowned for her pioneering work in mathematics and computing, often celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine. Her email address is ada.lovelace@example.com.
```

</hfoption>
<hfoption id="llama-index">

```python
query = "Tell me about Lady Ada Lovelace. What's her background?"
response = await alfred.run(query)

print("🎩 Alfred's Response:")
print(response.response.blocks[0].text)
```

Expected output:

```
🎩 Alfred's Response:
Lady Ada Lovelace was an English mathematician and writer, best known for her work on Charles Babbage's Analytical Engine. She was the first to recognize that the machine had applications beyond pure calculation.
```

</hfoption>
<hfoption id="langgraph">

```python
response = alfred.invoke({"messages": "Tell me about 'Lady Ada Lovelace'"})

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
```

Expected output:

```
🎩 Alfred's Response:
Ada Lovelace, also known as Augusta Ada King, Countess of Lovelace, was an English mathematician and writer. Born on December 10, 1815, and passing away on November 27, 1852, she is renowned for her work on Charles Babbage's Analytical Engine, a proposed mechanical general-purpose computer. Ada Lovelace is celebrated as one of the first computer programmers because she created a program for the Analytical Engine in 1843. She recognized that the machine could be used for more than mere calculation, envisioning its potential in a way that few did at the time. Her contributions to the field of computer science laid the groundwork for future developments. A day in October, designated as Ada Lovelace Day, honors women's contributions to science and technology, inspired by Lovelace's pioneering work.
```

</hfoption>
</hfoptions>


### Example 2: Checking the Weather for Fireworks

Let's see how Alfred can help us with the weather.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
query = "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```

Expected output (will vary due to randomness):
```
🎩 Alfred's Response:
I've checked the weather in Paris for you. Currently, it's clear with a temperature of 25°C. These conditions are perfect for the fireworks display tonight. The clear skies will provide excellent visibility for the spectacular show, and the comfortable temperature will ensure the guests can enjoy the outdoor event without discomfort.
```

</hfoption>
<hfoption id="llama-index">

```python
query = "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
response = await alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```

Expected output:

```
🎩 Alfred's Response:
The weather in Paris tonight is rainy with a temperature of 15°C. Given the rain, it may not be suitable for a fireworks display.
```

</hfoption>
<hfoption id="langgraph">

```python
response = alfred.invoke(
    {
        "messages": "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
    }
)

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
```

Expected output:

```
🎩 Alfred's Response:
The weather in Paris tonight is rainy with a temperature of 15°C, which may not be suitable for your fireworks display.
```
</hfoption>
</hfoptions>

### Example 3: Impressing AI Researchers

Let's see how Alfred can help us impress AI researchers.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
query = "One of our guests is from Qwen. What can you tell me about their most popular model?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```

Expected output:

```
🎩 Alfred's Response:
The most popular Qwen model is Qwen/Qwen2.5-VL-7B-Instruct with 3,313,345 downloads.
```
</hfoption>
<hfoption id="llama-index">

```python
query = "One of our guests is from Google. What can you tell me about their most popular model?"
response = await alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```

Expected output:

```
🎩 Alfred's Response:
The most popular model by Google on the Hugging Face Hub is google/electra-base-discriminator, with 28,546,752 downloads.
```

</hfoption>
<hfoption id="langgraph">

```python
response = alfred.invoke(
    {
        "messages": "One of our guests is from Qwen. What can you tell me about their most popular model?"
    }
)

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
```

Expected output:

```
🎩 Alfred's Response:
The most downloaded model by Qwen is Qwen/Qwen2.5-VL-7B-Instruct with 3,313,345 downloads.
```
</hfoption>
</hfoptions>

### Example 4: Combining Multiple Tools

Let's see how Alfred can help us prepare for a conversation with Dr. Nikola Tesla.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
query = "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```

Expected output:

```
🎩 Alfred's Response:
I've gathered information to help you prepare for your conversation with Dr. Nikola Tesla.

Guest Information:
Name: Dr. Nikola Tesla
Relation: old friend from university days
Description: Dr. Nikola Tesla is an old friend from your university days. He's recently patented a new wireless energy transmission system and would be delighted to discuss it with you. Just remember he's passionate about pigeons, so that might make for good small talk.
Email: nikola.tesla@gmail.com

Recent Advancements in Wireless Energy:
Based on my web search, here are some recent developments in wireless energy transmission:
1. Researchers have made progress in long-range wireless power transmission using focused electromagnetic waves
2. Several companies are developing resonant inductive coupling technologies for consumer electronics
3. There are new applications in electric vehicle charging without physical connections

Conversation Starters:
1. "I'd love to hear about your new patent on wireless energy transmission. How does it compare to your original concepts from our university days?"
2. "Have you seen the recent developments in resonant inductive coupling for consumer electronics? What do you think of their approach?"
3. "How are your pigeons doing? I remember your fascination with them."

This should give you plenty to discuss with Dr. Tesla while demonstrating your knowledge of his interests and recent developments in his field.
```

</hfoption>
<hfoption id="llama-index">

```python
query = "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
response = await alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```

Expected output:

```
🎩 Alfred's Response:
Here are some recent advancements in wireless energy that you might find useful for your conversation with Dr. Nikola Tesla:

1. **Advancements and Challenges in Wireless Power Transfer**: This article discusses the evolution of wireless power transfer (WPT) from conventional wired methods to modern applications, including solar space power stations. It highlights the initial focus on microwave technology and the current demand for WPT due to the rise of electric devices.

2. **Recent Advances in Wireless Energy Transfer Technologies for Body-Interfaced Electronics**: This article explores wireless energy transfer (WET) as a solution for powering body-interfaced electronics without the need for batteries or lead wires. It discusses the advantages and potential applications of WET in this context.

3. **Wireless Power Transfer and Energy Harvesting: Current Status and Future Trends**: This article provides an overview of recent advances in wireless power supply methods, including energy harvesting and wireless power transfer. It presents several promising applications and discusses future trends in the field.

4. **Wireless Power Transfer: Applications, Challenges, Barriers, and the
```

</hfoption>
<hfoption id="langgraph">

```python
response = alfred.invoke(
    {
        "messages": "I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"
    }
)

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
```

Expected output:

```
Based on the provided information, here are key points to prepare for the conversation with 'Dr. Nikola Tesla' about recent advancements in wireless energy:\n1. **Wireless Power Transmission (WPT):** Discuss how WPT revolutionizes energy transfer by eliminating the need for cords and leveraging mechanisms like inductive and resonant coupling.\n2. **Advancements in Wireless Charging:** Highlight improvements in efficiency, faster charging speeds, and the rise of Qi/Qi2 certified wireless charging solutions.\n3. **5G-Advanced Innovations and NearLink Wireless Protocol:** Mention these as developments that enhance speed, security, and efficiency in wireless networks, which can support advanced wireless energy technologies.\n4. **AI and ML at the Edge:** Talk about how AI and machine learning will rely on wireless networks to bring intelligence to the edge, enhancing automation and intelligence in smart homes and buildings.\n5. **Matter, Thread, and Security Advancements:** Discuss these as key innovations that drive connectivity, efficiency, and security in IoT devices and systems.\n6. **Breakthroughs in Wireless Charging Technology:** Include any recent breakthroughs or studies, such as the one from Incheon National University, to substantiate the advancements in wireless charging.
```
</hfoption>
</hfoptions>

## Advanced Features: Conversation Memory

To make Alfred even more helpful during the gala, we can enable conversation memory so he remembers previous interactions:

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
# Create Alfred with conversation memory
alfred_with_memory = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,
    planning_interval=3,
)

# First interaction
response1 = alfred_with_memory.run("Tell me about Lady Ada Lovelace.")
print("🎩 Alfred's First Response:")
print(response1)

# Second interaction (referencing the first)
response2 = alfred_with_memory.run("What projects is she currently working on?", reset=False)
print("🎩 Alfred's Second Response:")
print(response2)
```

</hfoption>
<hfoption id="llama-index">

```python
from llama_index.core.workflow import Context

alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool], llm=llm
)

# Remembering state
ctx = Context(alfred)

# First interaction
response1 = await alfred.run("Tell me about Lady Ada Lovelace.", ctx=ctx)
print("🎩 Alfred's First Response:")
print(response1)

# Second interaction (referencing the first)
response2 = await alfred.run("What projects is she currently working on?", ctx=ctx)
print("🎩 Alfred's Second Response:")
print(response2)
```

</hfoption>
<hfoption id="langgraph">

```python
# First interaction
response = alfred.invoke(
    {
        "messages": [
            HumanMessage(
                content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?"
            )
        ]
    }
)


print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
print()

# Second interaction (referencing the first)
response = alfred.invoke(
    {
        "messages": response["messages"]
        + [HumanMessage(content="What projects is she currently working on?")]
    }
)

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
```

</hfoption>
</hfoptions>

Notice that none of these three agent approaches directly couple memory with the agent. Is there a specific reason for this design choice 🧐?
* smolagents: Memory is not preserved across different execution runs, you must explicitly state it using `reset=False`.
* LlamaIndex: Requires explicitly adding a context object for memory management within a run.
* LangGraph: Offers options to retrieve previous messages or utilize a dedicated [MemorySaver](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot) component.

## Conclusion

Congratulations! You've successfully built Alfred, a sophisticated agent equipped with multiple tools to help host the most extravagant gala of the century. Alfred can now:

1. Retrieve detailed information about guests
2. Check weather conditions for planning outdoor activities
3. Provide insights about influential AI builders and their models
4. Search the web for the latest information
5. Maintain conversation context with memory

With these capabilities, Alfred is ready to ensure your gala is a resounding success, impressing guests with personalized attention and up-to-date information.

---

<!-- tools -->

# Building and Integrating Tools for Your Agent

In this section, we'll grant Alfred access to the web, enabling him to find the latest news and global updates. 
Additionally, he'll have access to weather data and Hugging Face hub model download statistics, so that he can make relevant conversation about fresh topics.

## Give Your Agent Access to the Web

Remember that we want Alfred to establish his presence as a true renaissance host, with a deep knowledge of the world.

To do so, we need to make sure that Alfred has access to the latest news and information about the world.

Let's start by creating a web search tool for Alfred!

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import DuckDuckGoSearchTool

# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchTool()

# Example usage
results = search_tool("Who's the current President of France?")
print(results)
```

Expected output:

```
The current President of France in Emmanuel Macron.
```


</hfoption>
<hfoption id="llama-index">

```python
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool

# Initialize the DuckDuckGo search tool
tool_spec = DuckDuckGoSearchToolSpec()

search_tool = FunctionTool.from_defaults(tool_spec.duckduckgo_full_search)
# Example usage
response = search_tool("Who's the current President of France?")
print(response.raw_output[-1]["body"])
```

Expected output:

```
The President of the French Republic is the head of state of France. The current President is Emmanuel Macron since 14 May 2017 defeating Marine Le Pen in the second round of the presidential election on 7 May 2017. List of French presidents (Fifth Republic) N° Portrait Name ...
```

</hfoption>
<hfoption id="langgraph">

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("Who's the current President of France?")
print(results)
```

Expected output:

```
Emmanuel Macron (born December 21, 1977, Amiens, France) is a French banker and politician who was elected president of France in 2017...
```

</hfoption>
</hfoptions>

## Creating a Custom Tool for Weather Information to Schedule the Fireworks

The perfect gala would have fireworks over a clear sky, we need to make sure the fireworks are not cancelled due to bad weather.

Let's create a custom tool that can be used to call an external weather API and get the weather information for a given location.

> [!TIP]
> For the sake of simplicity, we're using a dummy weather API for this example. If you want to use a real weather API, you could implement a weather tool that uses the OpenWeatherMap API, like in <a href="../../unit1/tutorial">Unit 1</a>.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import Tool
import random


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for.",
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # Dummy weather data
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20},
        ]
        # Randomly select a weather condition
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


# Initialize the tool
weather_info_tool = WeatherInfoTool()
```

</hfoption>
<hfoption id="llama-index">

```python
import random
from llama_index.core.tools import FunctionTool


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


# Initialize the tool
weather_info_tool = FunctionTool.from_defaults(get_weather_info)
```

</hfoption>
<hfoption id="langgraph">

```python
from langchain_core.tools import Tool
import random


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location.",
)
```

</hfoption>
</hfoptions>

## Creating a Hub Stats Tool for Influential AI Builders

In attendance at the gala are the who's who of AI builders. Alfred wants to impress them by discussing their most popular models, datasets, and spaces. We'll create a tool to fetch model statistics from the Hugging Face Hub based on a username.

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import Tool
from huggingface_hub import list_models


class HubStatsTool(Tool):
    name = "hub_stats"
    description = (
        "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    )
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from.",
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # List models from the specified author, sorted by downloads
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"


# Initialize the tool
hub_stats_tool = HubStatsTool()

# Example usage
print(hub_stats_tool("facebook"))  # Example: Get the most downloaded model by Facebook
```

Expected output:

```
The most downloaded model by facebook is facebook/esmfold_v1 with 12,544,550 downloads.
```

</hfoption>
<hfoption id="llama-index">

```python
import random
from llama_index.core.tools import FunctionTool
from huggingface_hub import list_models


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"


# Initialize the tool
hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)

# Example usage
print(hub_stats_tool("facebook"))  # Example: Get the most downloaded model by Facebook
```

Expected output:

```
The most downloaded model by facebook is facebook/esmfold_v1 with 12,544,550 downloads.
```

</hfoption>
<hfoption id="langgraph">

```python
from langchain_core.tools import Tool
from huggingface_hub import list_models


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"


# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub.",
)

# Example usage
print(hub_stats_tool.invoke("facebook"))  # Example: Get the most downloaded model by Facebook
```

Expected output:

```
The most downloaded model by facebook is facebook/esmfold_v1 with 13,109,861 downloads.
```

</hfoption>
</hfoptions>

With the Hub Stats Tool, Alfred can now impress influential AI builders by discussing their most popular models.

## Integrating Tools with Alfred

Now that we have all the tools, let's integrate them into Alfred's agent:

<hfoptions id="agents-frameworks">
<hfoption id="smolagents">

```python
from smolagents import CodeAgent, InferenceClientModel

# Initialize the Hugging Face model
model = InferenceClientModel()

# Create Alfred with all the tools
alfred = CodeAgent(tools=[search_tool, weather_info_tool, hub_stats_tool], model=model)

# Example query Alfred might receive during the gala
response = alfred.run("What is Facebook and what's their most popular model?")

print("🎩 Alfred's Response:")
print(response)
```

Expected output: 

```
🎩 Alfred's Response:
Facebook is a social networking website where users can connect, share information, and interact with others. The most downloaded model by Facebook on the Hugging Face Hub is ESMFold_v1.
```

</hfoption>
<hfoption id="llama-index">

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
# Create Alfred with all the tools
alfred = AgentWorkflow.from_tools_or_functions(
    [search_tool, weather_info_tool, hub_stats_tool], llm=llm
)

# Example query Alfred might receive during the gala
response = await alfred.run("What is Facebook and what's their most popular model?")

print("🎩 Alfred's Response:")
print(response)
```

Expected output: 

```
🎩 Alfred's Response:
Facebook is a social networking service and technology company based in Menlo Park, California. It was founded by Mark Zuckerberg and allows people to create profiles, connect with friends and family, share photos and videos, and join groups based on shared interests. The most popular model by Facebook on the Hugging Face Hub is `facebook/esmfold_v1` with 13,109,861 downloads.
```

</hfoption>
<hfoption id="langgraph">

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)


# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }


## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content="Who is Facebook and what's their most popular model?")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)
```

Expected output:

```
🎩 Alfred's Response:
Facebook is a social media company known for its social networking site, Facebook, as well as other services like Instagram and WhatsApp. The most downloaded model by Facebook on the Hugging Face Hub is facebook/esmfold_v1 with 13,202,321 downloads.
```
</hfoption>
</hfoptions>

## Conclusion

By integrating these tools, Alfred is now equipped to handle a variety of tasks, from web searches to weather updates and model statistics. This ensures he remains the most informed and engaging host at the gala.

> [!TIP]
> Try implementing a tool that can be used to get the latest news about a specific topic.
>
> When you're done, implement your custom tools in the <code>tools.py</code> file.

---

<!-- conclusion -->

# Conclusion

In this unit, we've learned how to create an agentic RAG system to help Alfred, our friendly neighborhood agent, prepare for and manage an extravagant gala.

The combination of RAG with agentic capabilities demonstrates how powerful AI assistants can become when they have:
- Access to structured knowledge (guest information)
- Ability to retrieve real-time information (web search)
- Domain-specific tools (weather information, Hub stats)
- Memory of past interactions

With these capabilities, Alfred is now well-equipped to be the perfect host, able to answer questions about guests, provide up-to-date information, and ensure the gala runs smoothly—even managing the perfect timing for the fireworks display!

> [!TIP]
> Now that you've built a complete agent, you might want to explore:
>
> - Creating more specialized tools for your own use cases
> - Implementing more sophisticated RAG systems with embeddings
> - Building multi-agent systems where agents can collaborate
> - Deploying your agent as a service that others can interact with
