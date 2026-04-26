### Agentic AI

A LLM is just a brain that responds to a `prompt` (an answer).  
A normal LLM call is one input → one output. 

An agent is a system that uses the LLM `repeatedly` to decide what to do next.  
Agents work in loops and can use tools (APIs, files, code), it's a process).  

~~~python
# LLM (one-shot)
response = llm("What is the weather in Paris?")
print(response)

# Agent (loop-based)
while True:
    thought = llm(context)

    if "need_weather" in thought:
        result = get_weather("Paris")
        context += result
    else:
        break
~~~

### Agent Loop

The loop is the heart of the agent.  
The agent thinks, act, observe, then `repeat`. 

~~~python
context = "User: Find best Python course"

while True:
    thought = llm(context)

    action = parse_action(thought)

    if action == "search":
        result = search_web()
    elif action == "finish":
        break

    context += f"Observation: {result}"
~~~

### Interaction

Agents don't jump to answers.  
They reason step-by-step and interact with the world.  
This is what makes agents feel intelligent.  

~~~python
# Example Flow
Thought: I need course options
Action: search("Python courses")
Observation: Found 3 courses

Thought: Compare them
Action: analyze(data)
Observation: Course A is best

Thought: Done

# Code Sketch
thought = llm(context)
action = extract_action(thought)
result = run_tool(action)

context += f"Observation: {result}"
~~~

### Tools / Function Calling

Tools let the agent interact with real systems (files, APIs, DBs).  
Without tools, the agent is stuck in text.  

~~~python
# Example Tool
def read_file(path):
    with open(path) as f:
        return f.read()

# Agent using tool
if "read_file" in thought:
    result = read_file("notes.md")
~~~

Better prompts improve answers.  
Tools enable actions, which is what agents are about. 

~~~python
# Prompt only
llm("Summarize my notes about Python")

# With tool
notes = read_file("python_notes.md")
llm(f"Summarize this: {notes}")
~~~