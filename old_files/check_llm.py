import ollama
 
question = "how to create a form in pml, Share me the syntax?"
 
prompt = f"""
You are an AVEVA E3D expert assistant.
Answer the below question.
 
Question:
{question}
"""
print("before stream")
stream = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}],
    stream=True
)
print("after stream")
print("\nANSWER:\n")
 
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)