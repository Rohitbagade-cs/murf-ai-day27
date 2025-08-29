import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

WEATHER_KEY = os.getenv("WEATHER_KEY")

print("🌤️ Weather API Debug")
print("=" * 30)

# Check API key
print("1. Checking API Key...")
if not WEATHER_KEY:
    print("❌ WEATHER_KEY not found in .env file!")
    print("\n📝 To fix this:")
    print("1. Go to https://openweathermap.org/")
    print("2. Sign up for free account")
    print("3. Get your API key")
    print("4. Add this line to your .env file:")
    print("   WEATHER_KEY=your_api_key_here")
    exit()
else:
    print(f"✅ API Key found: {WEATHER_KEY[:8]}...")

# Test API call
print("\n2. Testing API call to OpenWeather...")
test_city = "Pune"
url = f"http://api.openweathermap.org/data/2.5/weather?q={test_city}&appid={WEATHER_KEY}&units=metric"

try:
    response = requests.get(url, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        description = data['weather'][0]['description']
        print(f"✅ SUCCESS! Weather in {test_city}:")
        print(f"   🌡️ Temperature: {temp}°C")
        print(f"   ☁️ Condition: {description}")
        print("\n🎉 Your weather API is working!")
        
    elif response.status_code == 401:
        print("❌ UNAUTHORIZED (401)")
        print("Your API key is invalid or not activated yet.")
        print("New API keys can take 10-60 minutes to activate.")
        
    else:
        print(f"❌ ERROR {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n🔧 If you need an API key:")
print("Visit: https://openweathermap.org/api")