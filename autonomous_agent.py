import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from ollama import Client 

ollama_client = Client(host='http://127.0.0.1:11434')

# =============================================================================
# 1. TOOL DEFINITIONS
# =============================================================================

def search_web(query: str) -> str:
    """Searches DuckDuckGo HTML and returns a formatted string of results."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for a in soup.select('a.result__snippet'):
            results.append(a.text.strip())
            if len(results) >= 3: # Limit to top 3 snippets to save context
                break
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"

def read_url(url: str) -> str:
    """Fetches and extracts the main text content from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text(separator='\n', strip=True)
        return text[:2000] # Truncate to prevent context overflow
    except Exception as e:
        return f"Read error: {str(e)}"

# Map string tool names to actual functions
TOOLS = {
    "search_web": search_web,
    "read_url": read_url
}

# =============================================================================
# 2. AGENT LOGIC
# =============================================================================

SYSTEM_PROMPT = """You are an autonomous research agent. You must answer the user's query by gathering information using tools.
You must respond in STRICT JSON format with this exact structure:
{
    "thought": "Your reasoning about what to do next.",
    "tool": "search_web" or "read_url" or "finish",
    "input": "The argument for the tool, or your final answer if tool is 'finish'"
}

Rules:
- If you need information, use 'search_web'.
- If you have a URL from a search and need details, use 'read_url'.
- If you have enough information to answer the user's original query completely, use 'finish'.
- Do not include any text outside the JSON object."""

def run_agent(user_query: str, max_iterations: int = 6) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    final_answer = "Agent failed to produce an answer."

    for i in range(max_iterations):
        print(f"[Iteration {i+1}] Thinking...")
        response = ollama_client.chat(
            model='llama3.1:8b', 
            messages=messages,
            format='json'
        )
        
        raw_output = response['message']['content']
        
        try:
            action = json.loads(raw_output)
            tool_name = action.get("tool")
            tool_input = action.get("input")
        except json.JSONDecodeError:
            return "Error: Model failed to output valid JSON."

        print(f"Action: {tool_name}")

        if tool_name == "finish":
            final_answer = tool_input
            break
        elif tool_name in TOOLS:
            observation = TOOLS[tool_name](tool_input)
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            return f"Error: Unknown tool '{tool_name}'."
            
    return final_answer


# =============================================================================
# 3. EXECUTION
# =============================================================================

import gradio as gr

def agent_interface(query):
    if not query.strip():
        return "Please enter a valid query."
    # Run the agent and return the result to the UI
    return run_agent(query)

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 Local Autonomous Research Agent")
    gr.Markdown("Powered by Ollama (llama3.1:8b) and the ReAct framework.")
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Research Query", 
                placeholder="e.g., Compare the memory architecture of the M3 Max chip vs the RTX 4090.",
                lines=3
            )
            submit_btn = gr.Button("Run Agent", variant="primary")
        
        with gr.Column(scale=3):
            output_box = gr.Textbox(
                label="Agent's Final Answer", 
                lines=10,
                interactive=False
            )
    
    # Link the button to the function
    submit_btn.click(fn=agent_interface, inputs=query_input, outputs=output_box)

if __name__ == "__main__":
    # Launch the local web server
    demo.launch(server_name="127.0.0.1", server_port=7860)