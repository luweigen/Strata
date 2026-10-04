import json, sys
sys.path.insert(0, ".")
from serve.frontend import ChatTemplate, OutputParser, openai_to_messages
tools = [{"type": "function", "function": {"name": "cd", "description": "Change directory.",
          "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}, "required": ["folder"]}}},
         {"type": "function", "function": {"name": "ls", "description": "List files.",
          "parameters": {"type": "object", "properties": {"a": {"type": "boolean"}}}}}]
req = {"model": "x", "tools": tools, "chat_template_kwargs": {"enable_thinking": True}, "messages": [
    {"role": "system", "content": "You are a file assistant."},
    {"role": "user", "content": "Go to temp and list everything."},
    {"role": "assistant", "content": "", "reasoning_content": "I should cd into temp first.",
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "cd", "arguments": "{\"folder\": \"temp\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "{\"current_working_directory\": \"temp\"}"},
    {"role": "system", "content": "Note: hidden files matter."},          # mid-conversation system message
    {"role": "user", "content": "Now list with hidden files."}]}
messages, tls, kw = openai_to_messages(req)
out = {}
for name, path in (("strata default (pack)", "../Strata-data/packs/coder-iq1_m/tokenizer/chat_template.jinja"),
                   ("froggeric v22.4 (Halo)", sys.argv[1])):
    try:
        out[name] = ChatTemplate(path).render(messages, tools=tls, **kw)
        p = out[name]
        print(f"== {name}: renders OK, {len(p)} chars; earlier reasoning kept: {'cd into temp first' in p}; "
              f"mid system msg kept: {'hidden files matter' in p}; tool format: "
              f"{'XML <function=' if '<function=' in p else 'JSON' if '\"arguments\"' in p else '?'}")
        print("   ends with:", repr(p[-70:]))
    except Exception as e:
        print(f"== {name}: FAILED: {type(e).__name__}: {e}")
reply = "Listing now.\n</think>\n\n<tool_call>\n<function=ls>\n<parameter=a>\ntrue\n</parameter>\n</function>\n</tool_call>"
ps = OutputParser(thinking=True, tools=tls); evs = ps.feed(reply) + ps.finish()
print("parser on an XML reply:", [(e.kind, e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"])
import difflib
a, b = list(out.values())
for l in difflib.unified_diff(a.splitlines(), b.splitlines(), "strata default", "froggeric v22.4", n=0, lineterm=""):
    print(l[:200])
