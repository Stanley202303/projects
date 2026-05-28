from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import requests

GROQ_API_KEY = "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"
SYSTEM_PROMPT = "Keep replies clear and under 500 characters. Use plain text only."


def ask_groq(message: str, history=None):
    if not GROQ_API_KEY:
        return "Missing GROQ_API_KEY on Mac."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if isinstance(history, list):
        for item in history[-20:]:
            role = item.get("role", "user")
            content = str(item.get("content", ""))
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 140,
            },
            timeout=20,
        )
        data = response.json()
        if "choices" not in data:
            return "Groq error."
        reply = data["choices"][0]["message"]["content"]
        reply = reply.replace("\n", " ").replace("\r", " ").strip()
        return reply[:500]
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
            message = str(data.get("message", ""))
            history = data.get("history", [])
            reply = ask_groq(message, history)
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


if __name__ == "__main__":
    print("Pico AI server running on port 8080")
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
