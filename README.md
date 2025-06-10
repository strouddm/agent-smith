# Agent Smith 🤖

A conversational AI assistant powered by Google's Gemini model, capable of performing web searches and maintaining context-aware conversations.

## Features

- Natural language conversations with context awareness
- Web search integration using DuckDuckGo
- Powered by Google's Gemini 2.5 Pro model
- Smart query transformation for better search results
- Conversation history management
- Retry mechanism for failed operations
- Detailed logging for debugging

## Available Models

- gemini-2.5-pro-preview-06-05: Latest preview version with advanced reasoning and multimodal capabilities
- gemini-2.5-flash-preview-05-20: Preview model optimized for speed and cost efficiency
- gemini-1.5-pro: Stable version for complex tasks requiring high intelligence
- gemini-1.5-flash: Stable model for fast and versatile performance
- gemini-2.0-flash: Next-gen model focused on speed and real-time streaming
- gemini-2.0-flash-lite: Cost-effective model for high-throughput applications
- text-embedding-004: Specialized model for text embeddings and classification

## Prerequisites

- Python 3.8 or higher
- Google API key for Gemini
- Internet connection for web searches

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agent-smith.git
cd agent-smith
```

2. Create and activate a virtual environment:

For macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

For Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:

For macOS/Linux:
```bash
export GOOGLE_API_KEY="your_google_api_key"
```

For Windows (Command Prompt):
```bash
set GOOGLE_API_KEY=your_google_api_key
```

For Windows (PowerShell):
```bash
$env:GOOGLE_API_KEY="your_google_api_key"
```

## Usage

1. Ensure your virtual environment is activated (you should see `(venv)` in your terminal prompt)

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

4. Start chatting with Agent Smith! You can:
   - Ask general questions
   - Have contextual conversations

## Deactivating the Virtual Environment

When you're done using the application, you can deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```
agent-smith/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── agent_smith.jpg     # Agent Smith avatar
├── venv/              # Virtual environment directory (created during setup)
└── logs/              # Application logs directory
```

## Key Components

### Agent Class
- Handles message processing
- Manages conversation flow
- Integrates search functionality
- Uses LangGraph for workflow management

### SearchTool Class
- Performs web searches using DuckDuckGo
- Implements query transformation
- Includes rate limiting and retry mechanisms

### ChatInterface Class
- Manages the Streamlit UI
- Handles user interactions
- Displays messages and search results

## Error Handling

The application includes comprehensive error handling for:
- Search failures
- LLM errors
- Network issues
- Invalid inputs

## Logging

Logs are stored in the `logs` directory with timestamps, including:
- Application events
- Search operations
- Error messages
- State transitions