import ollama
messages="This biological sample has the following features: - Hydrophobicity: 0.42 (above average)- Charge: -1.0 PTM sites: 3  The model classified it as class 1. Exlain this result in plain English."

response = ollama.chat(model="llama2:latest", messages=[{"role": "user", "content": messages}])
reply = response.message.content
print(reply) 
