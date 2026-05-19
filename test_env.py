from dotenv import load_dotenv
load_dotenv()
import os
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))