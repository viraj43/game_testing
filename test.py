"""
Quick script to check available Gemini models
Run this to see what models work with your API key
"""

import google.generativeai as genai

# Replace with your actual API key
API_KEY = "AIzaSyCXipdk3XF5nX4shfnGPi2OyShX-s6DEsY"

genai.configure(api_key=API_KEY)

print("=" * 60)
print("AVAILABLE GEMINI MODELS FOR generateContent:")
print("=" * 60)

available_models = []

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✓ {model.name}")
        available_models.append(model.name)
        
        # Show details
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Input Token Limit: {model.input_token_limit}")
        print(f"  Output Token Limit: {model.output_token_limit}")
        print()

print("=" * 60)
print(f"Total available models: {len(available_models)}")
print("=" * 60)

# Try to use the first available model
if available_models:
    print(f"\nTesting first available model: {available_models[0]}")
    try:
        # Extract just the model name (remove 'models/' prefix if present)
        model_name = available_models[0].replace('models/', '')
        test_model = genai.GenerativeModel(model_name)
        response = test_model.generate_content("Say 'Hello World'")
        print(f"✅ SUCCESS! Model works: {model_name}")
        print(f"Response: {response.text}")
        print(f"\n🎯 USE THIS MODEL NAME IN YOUR CODE: '{model_name}'")
    except Exception as e:
        print(f"❌ Error testing model: {e}")
else:
    print("❌ No models found! Check your API key.")