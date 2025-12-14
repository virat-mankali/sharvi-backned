# Veda AI - Cursor-Style Data Analyst Agent

## Overview

Veda AI is a streaming data analyst agent that works like Cursor AI but for data analysis instead of coding. It analyzes machine production data from a PostgreSQL database with ~20 million rows, streaming its reasoning, SQL queries, and insights in real-time.

## Key Features

### 🎯 Cursor-Style Streaming
- Real-time token streaming as the AI thinks
- Visible reasoning process ("I'll query the database to find...")
- SQL queries shown before execution
- Progress updates during tool execution
- Results streamed as they arrive

### 🔄 ReAct Loop
The agent follows a Reason-Act-Observe loop:
1. **Reason**: Understand the question, plan the approach
2. **Act**: Execute SQL queries or export data
3. **Observe**: Analyze results, decide if more queries needed
4. **Repeat**: Continue until the answer is complete

### 📊 Automatic Visualizations
- Bar charts for comparisons
- Line charts for time series
- Pie charts for distributions
- Automatic chart type selection based on data

### 📥 CSV Export
- Large dataset export to Supabase Storage
- Signed download URLs (1 hour expiry)
- Server-side cursor for memory efficiency

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│  - SSE event listener                                        │
│  - Real-time message rendering                               │
│  - Chart.js visualizations                                   │
│  - Download handling                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ POST /chat/stream (SSE)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                            │
│  - SSE streaming endpoint                                    │
│  - Event formatting                                          │
│  - Error handling                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph ReAct Agent                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │  Agent  │───▶│  Tools  │───▶│  Agent  │──▶ ...           │
│  │ (Think) │    │ (Query) │    │(Analyze)│                  │
│  └─────────┘    └─────────┘    └─────────┘                  │
│                                                              │
│  Stream Modes: messages, updates, custom                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tools Layer                               │
│  - run_sql_query: Execute SQL, return max 100 rows          │
│  - export_to_csv: Export to Supabase Storage                │
│  - get_stream_writer: Send progress updates                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Supabase (PostgreSQL)                        │
│  - Table: "General Machine 2000"                            │
│  - ~20 million rows                                          │
│  - Storage bucket for CSV exports                           │
└─────────────────────────────────────────────────────────────┘
```

## Streaming Event Types

The `/chat/stream` endpoint sends these SSE events:

| Event Type | Description | Data |
|------------|-------------|------|
| `start` | Processing started | `{message, timestamp}` |
| `thinking` | Agent reasoning | `{content}` |
| `tool_call` | Tool being invoked | `{tool, args}` |
| `tool_start` | Tool execution starting | `{tool, message, sql}` |
| `tool_result` | Tool completed | `{tool, message, row_count, preview}` |
| `tool_error` | Tool failed | `{message}` |
| `token` | LLM token | `{content, node}` |
| `message` | Complete message chunk | `{content, node}` |
| `data_received` | Query results | `{row_count, preview}` |
| `visualization` | Chart data ready | `{data}` |
| `download` | CSV URL ready | `{url, row_count}` |
| `done` | Agent finished | `{answer, visualization, download_url}` |
| `error` | Error occurred | `{message}` |

## API Endpoints

### POST /chat/stream
Streaming endpoint with Server-Sent Events.

```javascript
const eventSource = new EventSource('/chat/stream', {
  method: 'POST',
  body: JSON.stringify({ query: "What's the best machine?", thread_id: "session-1" })
});

eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  const data = JSON.parse(event.data);
  // Handle different event types
  switch (data.type) {
    case 'token':
      appendToMessage(data.content);
      break;
    case 'visualization':
      renderChart(data.data);
      break;
    // ... etc
  }
};
```

### POST /chat
Non-streaming endpoint for simple requests.

```json
Request: { "query": "What's the best machine?", "thread_id": "session-1" }
Response: {
  "answer": "Based on the data...",
  "visualization": { "type": "bar", "labels": [...], "datasets": [...] },
  "download_url": null,
  "error": null
}
```

### GET /health
Health check endpoint.

## Environment Variables

```env
# Required
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...

# Optional
OPENAI_MODEL=gpt-4o          # or gpt-4o-mini for faster/cheaper
DATABASE_URL=postgresql://...  # Direct connection string
SUPABASE_DB_PASSWORD=...       # For building connection string
```

## Running Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

## Frontend Integration Example

```typescript
// React hook for streaming chat
function useStreamingChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = async (query: string) => {
    setIsStreaming(true);
    
    const response = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, thread_id: 'session-1' })
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            setIsStreaming(false);
            return;
          }
          
          const event = JSON.parse(data);
          handleEvent(event);
        }
      }
    }
  };

  return { messages, sendMessage, isStreaming };
}
```

## Design Decisions

### Why Streaming?
- Users see progress immediately (no waiting for complete response)
- Transparent reasoning builds trust
- SQL queries visible before execution
- Better UX for long-running analyses

### Why ReAct Pattern?
- Natural reasoning flow (think → act → observe)
- Can handle complex multi-step queries
- Self-correcting (can retry failed queries)
- Extensible (easy to add new tools)

### Why Multiple Stream Modes?
- `messages`: Token-by-token LLM output
- `updates`: Node completion events
- `custom`: Tool progress updates
- Combined for rich real-time experience

## Troubleshooting

### Streaming not working
- Check CORS settings
- Ensure `X-Accel-Buffering: no` header is set
- Verify SSE content type: `text/event-stream`

### SQL errors
- Table name must be quoted: `"General Machine 2000"`
- Check column names match schema
- Use COALESCE for NULL handling

### Slow responses
- Use `gpt-4o-mini` for faster responses
- Add appropriate LIMIT clauses
- Use aggregations instead of raw data
