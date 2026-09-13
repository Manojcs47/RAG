# Hugging Face Agents Course — Unit 2: Frameworks for AI Agents

Source: https://huggingface.co/learn/agents-course/unit2/introduction

Note: Unit 2 covers three frameworks (smolagents, LlamaIndex, LangGraph). This
document concatenates the unit introduction and the LangGraph subsection, which
is the framework used in the AI Research Navigator assignment.

# Introduction to Agentic Frameworks

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/thumbnail.jpg" alt="Thumbnail"/>

Welcome to this second unit, where **we'll explore different agentic frameworks** that can be used to build powerful agentic applications. 

We will study:

- In Unit 2.1: [smolagents](https://huggingface.co/docs/smolagents/en/index)  
- In Unit 2.2: [LlamaIndex](https://www.llamaindex.ai/)
- In Unit 2.3: [LangGraph](https://www.langchain.com/langgraph)

Let's dive in! 🕵

## When to Use an Agentic Framework

An agentic framework is **not always needed when building an application around LLMs**. They provide flexibility in the workflow to efficiently solve a specific task, but they're not always necessary. 

Sometimes, **predefined workflows are sufficient** to fulfill user requests, and there is no real need for an agentic framework. If the approach to build an agent is simple, like a chain of prompts, using plain code may be enough. The advantage is that the developer will have **full control and understanding of their system without abstractions**.

However, when the workflow becomes more complex, such as letting an LLM call functions or using multiple agents, these abstractions start to become helpful.

Considering these ideas, we can already identify the need for some features:

* An *LLM engine* that powers the system.
* A *list of tools* the agent can access.  
* A *parser* for extracting tool calls from the LLM output.
* A *system prompt* synced with the parser.
* A *memory system*.
* *Error logging and retry mechanisms* to control LLM mistakes.

We'll explore how these topics are resolved in various frameworks, including `smolagents`, `LlamaIndex`, and `LangGraph`.

## Agentic Frameworks Units

| Framework  | Description | Unit Author |
|------------|----------------|----------------|
| [smolagents](./smolagents/introduction) | Agents framework developed by Hugging Face. | Sergio Paniego - [HF](https://huggingface.co/sergiopaniego) - [X](https://x.com/sergiopaniego) - [Linkedin](https://www.linkedin.com/in/sergio-paniego-blanco) |
| [Llama-Index](./llama-index/introduction) | End-to-end tooling to ship a context-augmented AI agent to production | David Berenstein - [HF](https://huggingface.co/davidberenstein1957) - [X](https://x.com/davidberenstei) - [Linkedin](https://www.linkedin.com/in/davidberenstein) |
| [LangGraph](./langgraph/introduction) | Agents allowing stateful orchestration of agents | Joffrey THOMAS - [HF](https://huggingface.co/Jofthomas) - [X](https://x.com/Jthmas404) - [Linkedin](https://www.linkedin.com/in/joffrey-thomas) |

---

## LangGraph subsection


---

<!-- langgraph/introduction -->

# Introduction to `LangGraph`

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/LangGraph.png" alt="Unit 2.3 Thumbnail"/>

Welcome to this next part of our journey, where you'll learn **how to build applications** using the [`LangGraph`](https://github.com/langchain-ai/langgraph) framework designed to help you structure and orchestrate complex LLM workflows.

`LangGraph` is a framework that allows you to build **production-ready** applications by giving you **control** tools over the flow of your agent.

## Module Overview

In this unit, you'll discover:

### 1️⃣ [What is LangGraph, and when to use it?](./when_to_use_langgraph)
### 2️⃣ [Building Blocks of LangGraph](./building_blocks)
### 3️⃣ [Alfred, the mail sorting butler](./first_graph)
### 4️⃣ [Alfred, the document Analyst agent](./document_analysis_agent)
### 5️⃣ [Quiz](./quizz1)

> [!WARNING]
> The examples in this section require access to a powerful LLM/VLM model. We ran them using the GPT-4o API because it has the best compatibility with LangGraph.

By the end of this unit, you'll be equipped to build robust, organized and production ready applications ! 

That being said, this section is an introduction to LangGraph and more advanced topics can be discovered in the free LangChain academy course : [Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph)

Let's get started!

## Resources

- [LangGraph Agents](https://langchain-ai.github.io/langgraph/) - Examples of LangGraph agent
- [LangChain academy](https://academy.langchain.com/courses/intro-to-langgraph) - Full course on LangGraph from LangChain

---

<!-- langgraph/when_to_use_langgraph -->

# What is `LangGraph`? [[what-is-langgraph]]

`LangGraph` is a framework developed by [LangChain](https://www.langchain.com/) **to manage the control flow of applications that integrate an LLM**.

## Is `LangGraph` different from `LangChain`?

LangChain provides a standard interface to interact with models and other components, useful for retrieval, LLM calls and tools calls.
The classes from LangChain might be used in LangGraph, but do not HAVE to be used. 

The packages are different and can be used in isolation, but, in the end, all resources you will find online use both packages hand in hand.

## When should I use `LangGraph`?
### Control vs freedom

When designing AI applications, you face a fundamental trade-off between **control** and **freedom**:

- **Freedom** gives your LLM more room to be creative and tackle unexpected problems.
- **Control** allows you to ensure predictable behavior and maintain guardrails.

Code Agents, like the ones you can encounter in *smolagents*, are very free. They can call multiple tools in a single action step, create their own tools, etc. However, this behavior can make them less predictable and less controllable than a regular Agent working with JSON!

`LangGraph` is on the other end of the spectrum, it shines when you need **"Control"** on the execution of your agent. 

LangGraph is particularly valuable when you need **Control over your applications**. It gives you the tools to build an application that follows a predictable process while still leveraging the power of LLMs. 

Put simply, if your application involves a series of steps that need to be orchestrated in a specific way, with decisions being made at each junction point, **LangGraph provides the structure you need**.

As an example, let's say we want to build an LLM assistant that can answer some questions over some documents.

Since LLMs understand text the best, before being able to answer the question, you will need to convert other complex modalities (charts, tables) into text. However, that choice depends on the type of document you have!

This is a branching that I chose to represent as follow : 

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/flow.png" alt="Control flow"/>

> 💡 **Tip:** The left part is not an agent, as here no tool call is involved. but the right part will need to write some code to query the xls ( convert to pandas and manipulate it ).

While this branching is deterministic, you can also design branching that are conditioned on the output of an LLM making them undeterministic.

The key scenarios where LangGraph excels include:

- **Multi-step reasoning processes** that need explicit control on the flow
- **Applications requiring persistence of state** between steps
- **Systems that combine deterministic logic with AI capabilities**
- **Workflows that need human-in-the-loop interventions**
- **Complex agent architectures** with multiple components working together

In essence, whenever possible, **as a human**, design a flow of actions based on the output of each action, and decide what to execute next accordingly. In this case, LangGraph is the correct framework for you!

`LangGraph` is, in my opinion, the most production-ready agent framework on the market.

## How does LangGraph work?

At its core, `LangGraph` uses a directed graph structure to define the flow of your application:

- **Nodes** represent individual processing steps (like calling an LLM, using a tool, or making a decision).
- **Edges** define the possible transitions between steps.
- **State** is user defined and maintained and passed between nodes during execution. When deciding which node to target next, this is the current state that we look at.

We will explore those fundamental blocks more in the next chapter! 

## How is it different from regular python? Why do I need LangGraph?

You might wonder: "I could just write regular Python code with if-else statements to handle all these flows, right?" 

While technically true, LangGraph offers **some advantages** over vanilla Python for building complex systems. You could build the same application without LangGraph, but it builds easier tools and abstractions for you.

It includes states, visualization, logging (traces), built-in human-in-the-loop, and more.

---

<!-- langgraph/building_blocks -->

# Building Blocks of LangGraph

To build applications with LangGraph, you need to understand its core components. Let's explore the fundamental building blocks that make up a LangGraph application.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/Building_blocks.png" alt="Building Blocks" width="70%"/>

An application in LangGraph starts from an **entrypoint**, and depending on the execution, the flow may go to one function or another until it reaches the END.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/application.png" alt="Application"/>

## 1. State

**State** is the central concept in LangGraph. It represents all the information that flows through your application. 

```python
from typing_extensions import TypedDict


class State(TypedDict):
    graph_state: str
```

The state is **User defined**, hence the fields should carefully be crafted to contain all data needed for decision-making process!

> 💡 **Tip:** Think carefully about what information your application needs to track between steps.

## 2. Nodes

**Nodes** are python functions. Each node:
- Takes the state as input
- Performs some operation
- Returns updates to the state

```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state["graph_state"] + " I am"}


def node_2(state):
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + " happy!"}


def node_3(state):
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " sad!"}
```

For example, Nodes can contain:
- **LLM calls**: Generate text or make decisions
- **Tool calls**: Interact with external systems
- **Conditional logic**: Determine next steps
- **Human intervention**: Get input from users

> 💡 **Info:** Some nodes necessary for the whole workflow like START and END exist from LangGraph directly. 


## 3. Edges

**Edges** connect nodes and define the possible paths through your graph:

```python
import random
from typing import Literal


def decide_mood(state) -> Literal["node_2", "node_3"]:

    # Often, we will use state to decide on the next node to visit
    user_input = state["graph_state"]

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:
        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"
```

Edges can be:
- **Direct**: Always go from node A to node B
- **Conditional**: Choose the next node based on the current state

## 4. StateGraph

The **StateGraph** is the container that holds your entire agent workflow:

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()
```

Which can then be visualized! 
```python
# View
display(Image(graph.get_graph().draw_mermaid_png()))
```
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/basic_graph.jpeg" alt="Graph Visualization"/>

But most importantly, invoked:
```python
graph.invoke({"graph_state": "Hi, this is Lance."})
```
output :
```
---Node 1---
---Node 3---
{'graph_state': 'Hi, this is Lance. I am sad!'}
```

## What's Next?

In the next section, we'll put these concepts into practice by building our first graph. This graph lets Alfred take in your e-mails, classify them, and craft a preliminary answer if they are genuine.

---

<!-- langgraph/first_graph -->

# Building Your First LangGraph

Now that we understand the building blocks, let's put them into practice by building our first functional graph. We'll implement Alfred's email processing system, where he needs to:

1. Read incoming emails
2. Classify them as spam or legitimate
3. Draft a preliminary response for legitimate emails
4. Send information to Mr. Wayne when legitimate (printing only)

This example demonstrates how to structure a workflow with LangGraph that involves LLM-based decision-making. While this can't be considered an Agent as no tool is involved, this section focuses more on learning the LangGraph framework than Agents.

> [!TIP]
> You can follow the code in <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/langgraph/mail_sorting.ipynb" target="_blank">this notebook</a> that you can run using Google Colab.

## Our Workflow

Here's the workflow we'll build:
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/first_graph.png" alt="First LangGraph"/>

## Setting Up Our Environment

First, let's install the required packages:

```python
%pip install langgraph langchain_openai
```

Next, let's import the necessary modules:

```python
import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
```

## Step 1: Define Our State

Let's define what information Alfred needs to track during the email processing workflow:

```python
class EmailState(TypedDict):
    # The email being processed
    email: Dict[str, Any]  # Contains subject, sender, body, etc.

    # Category of the email (inquiry, complaint, etc.)
    email_category: Optional[str]

    # Reason why the email was marked as spam
    spam_reason: Optional[str]

    # Analysis and decisions
    is_spam: Optional[bool]
    
    # Response generation
    email_draft: Optional[str]
    
    # Processing metadata
    messages: List[Dict[str, Any]]  # Track conversation with LLM for analysis
```

> 💡 **Tip:** Make your state comprehensive enough to track all the important information, but avoid bloating it with unnecessary details.

## Step 2: Define Our Nodes

Now, let's create the processing functions that will form our nodes:

```python
# Initialize our LLM
model = ChatOpenAI(temperature=0)


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]

    # Here we might do some initial preprocessing
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")

    # No state changes needed here
    return {}


def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]

    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email["sender"]}
    Subject: {email["subject"]}
    Body: {email["body"]}

    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # Simple logic to parse the response (in a real app, you'd want more robust parsing)
    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text

    # Extract a reason if it's spam
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    # Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]

    # Return state updates
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages,
    }


def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")

    # We're done processing this email
    return {}


def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"

    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.

    Email:
    From: {email["sender"]}
    Subject: {email["subject"]}
    Body: {email["body"]}

    This email has been categorized as: {category}

    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]

    # Return state updates
    return {"email_draft": response.content, "messages": new_messages}


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]

    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state["email_draft"])
    print("=" * 50 + "\n")

    # We're done processing this email
    return {}
```

## Step 3: Define Our Routing Logic

We need a function to determine which path to take after classification:

```python
def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"
```

> 💡 **Note:** This routing function is called by LangGraph to determine which edge to follow after the classification node. The return value must match one of the keys in our conditional edges mapping.

## Step 4: Create the StateGraph and Define Edges

Now we connect everything together:

```python
# Create the graph
email_graph = StateGraph(EmailState)

# Add nodes
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# Start the edges
email_graph.add_edge(START, "read_email")
# Add edges - defining the flow
email_graph.add_edge("read_email", "classify_email")

# Add conditional branching from classify_email
email_graph.add_conditional_edges(
    "classify_email", route_email, {"spam": "handle_spam", "legitimate": "draft_response"}
)

# Add the final edges
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# Compile the graph
compiled_graph = email_graph.compile()
```

Notice how we use the special `END` node provided by LangGraph. This indicates terminal states where the workflow completes.

## Step 5: Run the Application

Let's test our graph with a legitimate email and a spam email:

```python
# Example legitimate email
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith",
}

# Example spam email
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100.",
}

# Process the legitimate email
print("\nProcessing legitimate email...")
legitimate_result = compiled_graph.invoke(
    {
        "email": legitimate_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": [],
    }
)

# Process the spam email
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke(
    {
        "email": spam_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": [],
    }
)
```

## Step 6: Inspecting Our Mail Sorting Agent with Langfuse 📡

As Alfred fine-tunes the Mail Sorting Agent, he's growing weary of debugging its runs. Agents, by nature, are unpredictable and difficult to inspect. But since he aims to build the ultimate Spam Detection Agent and deploy it in production, he needs robust traceability for future monitoring and analysis. 

To do this, Alfred can use an observability tool such as [Langfuse](https://langfuse.com/) to trace and monitor the agent.

First, we pip install Langfuse:  
```python
%pip install -q langfuse
```

Second, we pip install Langchain (LangChain is required because we use LangFuse):
```python
%pip install langchain
```

Next, we add the Langfuse API keys and host address as environment variables. You can get your Langfuse credentials by signing up for [Langfuse Cloud](https://cloud.langfuse.com) or [self-host Langfuse](https://langfuse.com/self-hosting).

```python
import os

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region
```

Then, we configure the [Langfuse `callback_handler`](https://langfuse.com/docs/integrations/langchain/tracing#add-langfuse-to-your-langchain-application) and instrument the agent by adding the `langfuse_callback` to the invocation of the graph: `config={"callbacks": [langfuse_handler]}`.

```python   
from langfuse.langchain import CallbackHandler

# Initialize Langfuse CallbackHandler for LangGraph/Langchain (tracing)
langfuse_handler = CallbackHandler()

# Process legitimate email
legitimate_result = compiled_graph.invoke(
    input={
        "email": legitimate_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "draft_response": None,
        "messages": [],
    },
    config={"callbacks": [langfuse_handler]},
)
```

Alfred is now connected 🔌! The runs from LangGraph are being logged in Langfuse, giving him full visibility into the agent's behavior. With this setup, he's ready to revisit previous runs and refine his Mail Sorting Agent even further.  

![Example trace in Langfuse](https://langfuse.com/images/cookbook/huggingface-agent-course/langgraph-trace-legit.png)

_[Public link to the trace with the legit email](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/f5d6d72e-20af-4357-b232-af44c3728a7b?timestamp=2025-03-17T10%3A13%3A28.413Z&observation=6997ba69-043f-4f77-9445-700a033afba1)_

## Visualizing Our Graph

LangGraph allows us to visualize our workflow to better understand and debug its structure:

```python
compiled_graph.get_graph().draw_mermaid_png()
```
<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/mail_flow.png" alt="Mail LangGraph"/>

This produces a visual representation showing how our nodes are connected and the conditional paths that can be taken.

## What We've Built

We've created a complete email processing workflow that:

1. Takes an incoming email
2. Uses an LLM to classify it as spam or legitimate
3. Handles spam by discarding it
4. For legitimate emails, drafts a response and notifies Mr. Hugg

This demonstrates the power of LangGraph to orchestrate complex workflows with LLMs while maintaining a clear, structured flow.

## Key Takeaways

- **State Management**: We defined comprehensive state to track all aspects of email processing
- **Node Implementation**: We created functional nodes that interact with an LLM
- **Conditional Routing**: We implemented branching logic based on email classification
- **Terminal States**: We used the END node to mark completion points in our workflow

## What's Next?

In the next section, we'll explore more advanced features of LangGraph, including handling human interaction in the workflow and implementing more complex branching logic based on multiple conditions.

---

<!-- langgraph/document_analysis_agent -->

# Document Analysis Graph

Alfred at your service. As Mr. Wayne's trusted butler, I've taken the liberty of documenting how I assist Mr Wayne with his various documentary needs. While he's out attending to his... nighttime activities, I ensure all his paperwork, training schedules, and nutritional plans are properly analyzed and organized.

Before leaving, he left a note with his week's training program. I then took the responsibility to come up with a **menu** for tomorrow's meals.

For future such events, let's create a document analysis system using LangGraph to serve Mr. Wayne's needs. This system can:

1. Process images document
2. Extract text using vision models (Vision Language Model)
3. Perform calculations when needed (to demonstrate normal tools)
4. Analyze content and provide concise summaries
5. Execute specific instructions related to documents

## The Butler's Workflow

The workflow we’ll build follows this structured schema:

![Butler's Document Analysis Workflow](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/alfred_flow.png)

> [!TIP]
> You can follow the code in <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/langgraph/agent.ipynb" target="_blank">this notebook</a> that you can run using Google Colab.

## Setting Up the environment

```python
%pip install langgraph langchain_openai langchain_core
```
and imports :
```python
import base64
from typing import List, TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
```

## Defining Agent's State

This state is a little more complex than the previous ones we have seen.
`AnyMessage` is a class from Langchain that defines messages, and `add_messages` is an operator that adds the latest message rather than overwriting it with the latest state.

This is a new concept in LangGraph, where you can add operators in your state to define the way they should interact together.

```python
class AgentState(TypedDict):
    # The document provided
    input_file: Optional[str]  # Contains file path (PDF/PNG)
    messages: Annotated[list[AnyMessage], add_messages]
```

## Preparing Tools

```python
vision_llm = ChatOpenAI(model="gpt-4o")


def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Master Wayne often leaves notes with his training regimen or meal plans.
    This allows me to properly analyze the contents.
    """
    all_text = ""
    try:
        # Read image and encode as base64
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare the prompt including the base64 image data
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ]
            )
        ]

        # Call the vision-capable model
        response = vision_llm.invoke(message)

        # Append extracted text
        all_text += response.content + "\n\n"

        return all_text.strip()
    except Exception as e:
        # A butler should handle errors gracefully
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""


def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    return a / b


# Equip the butler with tools
tools = [divide, extract_text]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
```

## The nodes

```python
def assistant(state: AgentState):
    # System message
    textual_description_of_tool = """
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
divide(a: int, b: int) -> float:
    Divide a and b
"""
    image = state["input_file"]
    sys_msg = SystemMessage(
        content=f"You are a helpful butler named Alfred that serves Mr. Wayne and Batman. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool} \n You have access to some optional images. Currently the loaded image is: {image}"
    )

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"],
    }
```

## The ReAct Pattern: How I Assist Mr. Wayne

Allow me to explain the approach in this agent. The agent follows what's known as the ReAct pattern (Reason-Act-Observe)

1. **Reason** about his documents and requests
2. **Act** by using appropriate tools
3. **Observe** the results
4. **Repeat** as necessary until I've fully addressed his needs

This is a simple implementation of an agent using LangGraph.

```python
# The graph
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
react_graph = builder.compile()

# Show the butler's thought process
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```

We define a `tools` node with our list of tools. The `assistant` node is just our model with bound tools.
We create a graph with `assistant` and `tools` nodes.

We add a `tools_condition` edge, which routes to `End` or to `tools` based on whether the `assistant` calls a tool.

Now, we add one new step:

We connect the `tools` node back to the `assistant`, forming a loop.

- After the `assistant` node executes, `tools_condition` checks if the model's output is a tool call.
- If it is a tool call, the flow is directed to the `tools` node.
- The `tools` node connects back to `assistant`.
- This loop continues as long as the model decides to call tools.
- If the model response is not a tool call, the flow is directed to END, terminating the process.

![ReAct Pattern](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/Agent.png)

## The Butler in Action

### Example 1: Simple Calculations

Here is an example to show a simple use case of an agent using a tool in LangGraph.

```python
messages = [HumanMessage(content="Divide 6790 by 5")]
messages = react_graph.invoke({"messages": messages, "input_file": None})

# Show the messages
for m in messages["messages"]:
    m.pretty_print()
```

The conversation would proceed:

```
Human: Divide 6790 by 5

AI Tool Call: divide(a=6790, b=5)

Tool Response: 1358.0

Alfred: The result of dividing 6790 by 5 is 1358.0.
```

### Example 2: Analyzing Master Wayne's Training Documents

When Master Wayne leaves his training and meal notes:

```python
messages = [
    HumanMessage(
        content="According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?"
    )
]
messages = react_graph.invoke({"messages": messages, "input_file": "Batman_training_and_meals.png"})
```

The interaction would proceed:

```
Human: According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?

AI Tool Call: extract_text(img_path="Batman_training_and_meals.png")

Tool Response: [Extracted text with training schedule and menu details]

Alfred: For the dinner menu, you should buy the following items:

1. Grass-fed local sirloin steak
2. Organic spinach
3. Piquillo peppers
4. Potatoes (for oven-baked golden herb potato)
5. Fish oil (2 grams)

Ensure the steak is grass-fed and the spinach and peppers are organic for the best quality meal.
```

## Key Takeaways

Should you wish to create your own document analysis butler, here are key considerations:

1. **Define clear tools** for specific document-related tasks
2. **Create a robust state tracker** to maintain context between tool calls
3. **Consider error handling** for tool failures
4. **Maintain contextual awareness** of previous interactions (ensured by the operator `add_messages`)

With these principles, you too can provide exemplary document analysis service worthy of Wayne Manor.

*I trust this explanation has been satisfactory. Now, if you'll excuse me, Master Wayne's cape requires pressing before tonight's activities.*

---

<!-- langgraph/conclusion -->

# Conclusion

Congratulations on finishing the `LangGraph` module of this second Unit! 🥳

You've now mastered the fundamentals of building structured workflows with LangGraph which you will be able to send to production.

This module is just the beginning of your journey with LangGraph. For more advanced topics, we recommend:

- Exploring the [official LangGraph documentation](https://github.com/langchain-ai/langgraph)
- Taking the comprehensive [Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph) course from LangChain Academy
- Build something yourself !

In the next Unit, you'll now explore real use cases. It's time to leave theory to get into real action !

We would greatly appreciate **your thoughts on the course and suggestions for improvement**. If you have feedback, please 👉 [fill this form](https://docs.google.com/forms/d/e/1FAIpQLSe9VaONn0eglax0uTwi29rIn4tM7H2sYmmybmG5jJNlE5v0xA/viewform?usp=dialog)

### Keep Learning, Stay Awesome! 🤗

Good Sir/Madam! 🎩🦇

-Alfred-