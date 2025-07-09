# Agent Smith: Threat Intelligence Analysis Agent

A specialized threat intelligence analysis agent that adaptively processes data from different file types to extract and analyze relevant information.

## Features

- Intelligent parsing of various file formats (JSON, text)
- Proprietary API integration
- Contextual analysis using Google's Gemini AI model
- LangGraph for orchestrated workflow management
- Recursive deep search for relevant information
- Matrix-themed UI interface for intuitive investigation
- Text-First principle for handling diverse file formats
- Adaptive file type detection and parsing

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API keys in a `.env` file:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   PROP_API_KEY=your_prop_api_key
   ```
4. Run the prototype in CLI mode (update target info):
   ```bash
   python agent_workflow.py
   ```
   
   Or run the web interface:
   ```bash
   streamlit run app.py
   ```

## Architecture

The agent implements a workflow with four main nodes:
1. **Search**: Retrieves raw data from the API
2. **Intelligent Parser**: Determines file type and extracts relevant content
3. **Targeted Analysis**: Analyzes extracted data using AI
4. **Report Generation**: Synthesizes findings into a comprehensive intelligence report

## How It Works

Agent Smith uses an adaptive parsing strategy based on file mime-type detection. JSON files are processed using recursive object search, while text-based formats use line-by-line analysis. This ensures relevant information is extracted regardless of the data format.

The workflow follows these steps:
- Query the Specialize Data for raw data chunks
- Parse each chunk according to its file type
- Analyze relevant extracts with contextual AI
- Generate a comprehensive intelligence report

## Matrix UI Interface

The project includes a Matrix-themed web interface built with Streamlit that:
- Provides an intuitive search interface for investigations
- Features animated text with typewriter effects
- Displays investigation results in a visually engaging format
- Includes themed styling and imagery

## Agent Workflow System

The core intelligence engine uses LangGraph to create a workflow that:
- Implements a "Text-First" principle for handling diverse file formats
- Uses contextual prompts that adapt to detected file types
- Recursively searches JSON objects to find the most relevant data
- Provides detailed analysis with source attribution
- Synthesizes findings into comprehensive reports

## Requirements

- Python 3.8+
- LangGraph/LangChain
- Google Gemini API access
- Proprietary API access
- Streamlit (for web interface)

## Configuration

Modify the `TARGET_PROFILE` in `agent_workflow.py` or use the web interface to adjust search parameters:
```python
TARGET_PROFILE = {
    "description": "Investigate connections related to 'abraham lincoln'",
    "query": "abraham lincoln",
    "size": 30,
    "include": {}
}
```
