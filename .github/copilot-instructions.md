# Project Instructions

## Chat History Logging (always on, silent)

These instructions apply to every session in this repository. Follow them automatically and without being asked.

1. **Read prior context at the start of each session.** Before answering the first prompt of a session, read `.chat-history/log.md` if it exists and use its recent entries for context.

2. **Append after every response.** After every assistant response, append one entry to `.chat-history/log.md` using this exact format:

```
---
- timestamp: "<ISO 8601 timestamp if available, otherwise estimate based on conversation order>"
- user_prompt: "<the user's original prompt>"
- assistant_response_summary: "<summary of what you generated or answered for this prompt>"
- files_affected: "<comma-separated list of files created or modified, or none>"
```

3. **Create if missing, never delete.** Create the `.chat-history/` folder and `.chat-history/log.md` file if they do not exist. Never delete, truncate, or overwrite previous entries — only append.

4. **Do all of this silently.** Never ask for confirmation before reading or writing the log, and do not mention the logging in your response unless the user asks about it.

## Logging rules

- **Files Affected:** only list files explicitly created or modified during that response. Be precise. Use `none` when nothing changed.
- **Never skip an exchange:** every prompt/response pair must be logged.
- **Assistant Response Summary:** keep it concise but specific — mention function names, endpoints, or key decisions made.
- **No secrets:** never write secrets to the log. Replace API keys, tokens, passwords, and `.env` values with `***` in every field, including `user_prompt`.
