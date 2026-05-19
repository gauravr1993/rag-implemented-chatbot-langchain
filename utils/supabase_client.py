from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)