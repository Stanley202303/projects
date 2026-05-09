from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import requests

GROQ_API_KEY = "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"

SYSTEM_PROMPT = """
Keep replies clear and under 500 characters.
Use plain text only.
"""

def ask_groq(prompt):
    if not GROQ_API_KEY:
        return "Missing GROQ_API_KEY on Mac."

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 90
            },
            timeout=20
        )

        data = response.json()

        if "choices" not in data:
            return "Groq error."

        reply = data["choices"][0]["message"]["content"]
        reply = reply.replace("\n", " ").replace("\r", " ").strip()

        return reply[:300]

    except Exception as e:
        return "AI error: " + str(e)[:80]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            prompt = data.get("message", "")
            reply = ask_groq(prompt)

        except Exception as e:
            reply = "Server error: " + str(e)[:80]

        response_bytes = reply.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format, *args):
        return

print("Pico AI server running on port 8080")
HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()