import os
import sys

try:
    from dotenv import load_dotenv
    import google.generativeai as genai
except ImportError:
    print("Dependencies not installed. Please run: pip install -r requirements.txt")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key or api_key.startswith("AIzaSyPasteYour"):
    print("ERROR: GOOGLE_API_KEY is missing or invalid in your .env file.")
    sys.exit(1)

print("Authenticating with Google Gemini API...\n")
genai.configure(api_key=api_key)

try:
    models = genai.list_models()
    print("Available Models for your API Key:")
    print("-" * 50)
    found_any = False
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            found_any = True
            print(f"- {m.name.replace('models/', '')}")
    print("-" * 50)
    
    if not found_any:
        print("Your API key is valid, but you don't have access to any generateContent models.")
    else:
        print("\nFix: Copy one of the names above (e.g. 'gemini-1.5-flash-latest' or 'gemini-2.0-flash')")
        print("and paste it into your .env file for GENAI_MODEL_NAME.")
        
except Exception as e:
    print(f"Failed to fetch models: {e}")
