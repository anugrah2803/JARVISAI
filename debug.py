from dotenv import load_dotenv
load_dotenv()
import os
keys = []
first = os.getenv("GROQ_API_KEY", "").strip()
if first: keys.append(first)
i = 2
while True:
    k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
    if not k: break
    print(f"Found key {i}: {k[:20]}")
    keys.append(k)
    i += 1
print(f"Total: {len(keys)}")
