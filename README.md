# Agent Smith: Threat Intelligence Analysis Agent

A specialized threat intelligence analysis agent that adaptively processes data from different file types to extract and analyze relevant information.

## Features

- Intelligent parsing of various file formats (JSON, text)
- Strategic Entity Database (SED) API integration
- Contextual analysis using Google's Gemini AI model
- LangGraph for orchestrated workflow management
- Recursive deep search for relevant information

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API keys in a `.env` file:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   SED_API_KEY=your_sed_api_key
   ```
4. Run the prototype:
   ```bash
   python prototype.py
   ```

## Architecture

The agent implements a workflow with four main nodes:
1. **Search**: Retrieves raw data from the Flashpoint API
2. **Intelligent Parser**: Determines file type and extracts relevant content
3. **Targeted Analysis**: Analyzes extracted data using AI
4. **Report Generation**: Synthesizes findings into a comprehensive intelligence report

## How It Works

Agent Smith uses an adaptive parsing strategy based on file mime-type detection. JSON files are processed using recursive object search, while text-based formats use line-by-line analysis. This ensures relevant information is extracted regardless of the data format.

The workflow follows these steps:
- Query the Strategic Entity Database for raw data chunks
- Parse each chunk according to its file type
- Analyze relevant extracts with contextual AI
- Generate a comprehensive intelligence report

## Requirements

- Python 3.8+
- LangGraph/LangChain
- Google Gemini API access
- Flashpoint Strategic Entity Database (SED) API access

## Configuration

Modify the `TARGET_PROFILE` in `prototype.py` to adjust search parameters:
```python
TARGET_PROFILE = {
    "description": "Investigate connections related to 'mark zuckerberg'",
    "query": "mark zuckerberg",
    "size": 10,
    "include": {}
}
```