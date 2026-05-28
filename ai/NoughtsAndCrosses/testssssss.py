# import requests, time
# GROQ_API_KEY = "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"

# def get(board):
#     SYSTEM_PROMPT = f'''You are a noughts and crosses bot. Keep the reply as one digit between 0 and 8 inclusive, which is your move. Here is the board:
# 0 1 2
# 3 4 5
# 6 7 8
# The current state of you will be given in a format like this: '1,1,0,-1,-1,1,-1,1,0', starting from (0, 0) on the board (top left) and ending at bottom right (2, 2)
# In the example I gave you, the board would therefore be:
#  1  1  0
# -1 -1  1
# -1  1  0
# Make the best decision. You are -1 on the board and your opponent is 1. 
# The state of the board right now is {board}. What is the best move?
# '''
#     messages = [{"role": "system", "content": SYSTEM_PROMPT}]
#     response = requests.post(
#                 "https://api.groq.com/openai/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {GROQ_API_KEY}",
#                     "Content-Type": "application/json",
#                 },
#                 json={
#                     "model": "llama-3.1-8b-instant",
#                     "messages": messages,
#                     "temperature": 0,
#                     "max_tokens": 50,
#                 },
#                 timeout=20,
#             )
#     data = response.json()
#     if "choices" not in data:
#         return "Groq error."
#     reply = data["choices"][0]["message"]["content"]
#     reply = reply.replace("\n", " ").replace("\r", " ").strip()
#     time.sleep(2.05)
#     return reply
# print(get("1,1,0,-1,-1,1,-1,1,0"))
def best_move(board):
    board = list(map(int, board.split(",")))
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]

    # Win if possible
    for a, b, c in wins:
        line = [board[a], board[b], board[c]]
        if line.count(-1) == 2 and line.count(0) == 1:
            return [a, b, c][line.index(0)]

    # Block opponent if needed
    for a, b, c in wins:
        line = [board[a], board[b], board[c]]
        if line.count(1) == 2 and line.count(0) == 1:
            return [a, b, c][line.index(0)]

    # Otherwise pick center, corner, then side
    for i in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
        if board[i] == 0:
            return i

print(best_move("1,1,0,-1,-1,1,-1,1,0"))