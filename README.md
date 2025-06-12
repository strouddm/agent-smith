# Agent Smith 🤖

A conversational AI assistant powered by Google's Gemini model with advanced intelligence capabilities including web search and Strategic Entity Database (SED) integration.

## Features

- Natural language conversations with context awareness
- Web search integration using DuckDuckGo
- Strategic Entity Database (SED) integration for deep intelligence searches
- Automatic query optimization for more effective intelligence searches
- Document evaluation and scoring system for relevance and insight value
- SQLite database for persistent storage of search results and evaluations
- Context-aware search decision and query transformation for better results
- Powered by Google's Gemini 2.5 Pro model
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
- SED API key for intelligence database access
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
export SED_API_KEY="your_sed_api_key"
```

For Windows (Command Prompt):
```bash
set GOOGLE_API_KEY=your_google_api_key
set SED_API_KEY=your_sed_api_key
```

For Windows (PowerShell):
```bash
$env:GOOGLE_API_KEY="your_google_api_key"
$env:SED_API_KEY="your_sed_api_key"
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
   - Perform intelligence research on entities and topics
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
├── orchestrator.py     # Main orchestrator agent
├── sed_agent.py        # Strategic Entity Database agent
├── tools.py            # Tool implementations for search and SED
├── requirements.txt    # Project dependencies
├── sed_documents.db    # SQLite database for document storage
├── venv/               # Virtual environment directory (created during setup)
├── images/             # Images directory
├── logs/               # Application logs directory
└── archive/            # Old code
```

## Key Components

### OrchestratorAgent Class
- Main controller that routes requests
- Manages conversation flow
- Determines when to use web search vs. SED search
- Uses LangGraph for workflow management

### SEDAgent Class
- Specialized agent for intelligence database queries
- Implements multi-step workflow:
  - Query optimization for keyword-based searches
  - Document retrieval from SED API
  - Document storage in SQLite database
  - Document evaluation for relevance and insights
  - Result synthesis into comprehensive reports

### SearchTool Classes
- WebSearchTool: Performs web searches using DuckDuckGo
- SEDSearchTool: Interfaces with intelligence database API
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
- Database issues
- Invalid inputs

## Logging

Logs are stored in the `logs` directory with timestamps, including:
- Application events
- Search operations
- Database operations
- Document evaluations
- Error messages
- State transitions

## Database Schema

The application uses a SQLite database with two main tables:

**Documents Table:**
- doc_id: Unique identifier for each document
- query: The query that retrieved this document
- title: Document title
- content: Full document content in JSON format
- metadata: Additional document metadata
- created_at: Timestamp of when the document was stored

**Evaluations Table:**
- evaluation_id: Unique identifier for each evaluation
- doc_id: Foreign key to the documents table
- query: The query used for evaluation
- relevance_score: Numerical score (0-10) for document relevance
- insight_score: Numerical score (0-10) for insight value
- evaluation_text: Text explanation of the evaluation
- created_at: Timestamp of when the evaluation was performed