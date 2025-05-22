import speech_recognition as sr
import pyttsx3
import wikipedia
import requests
import webbrowser
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Weather API configuration (Get your free API key from OpenWeatherMap)
WEATHER_API_KEY = "YOUR_API_KEY"
CITY_NAME = "Thiruvananthapuram"


# Set female voice if available
voices = engine.getProperty('voices')

# for voice in voices:
#     print(f"ID: {voice.id} | Name: {voice.name} | Lang: {voice.languages}")

# female_voices = [v for v in voices if 'female' in v.name.lower() or 'zira' in v.name.lower()]
female_voices = [v for v in voices if 'female' in v.name.lower() or 'hazel' in v.name.lower()]

if female_voices:
    engine.setProperty('voice', female_voices[0].id)
else:
    print("Warning: No female voice found - using default voice")

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listen to microphone input and convert to text"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        
    try:
        query = r.recognize_google(audio).lower()
        print(f"User said: {query}")
        return query
    except Exception as e:
        speak("Sorry, I didn't catch that. Can you repeat?")
        return ""

def get_weather():
    """Get weather information for Thiruvananthapuram"""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={WEATHER_API_KEY}&q={CITY_NAME}&units=metric"
    
    try:
        response = requests.get(complete_url)
        data = response.json()
        
        if data["cod"] != "404":
            main = data["main"]
            temperature = main["temp"]
            humidity = main["humidity"]
            weather_desc = data["weather"][0]["description"]
            
            report = (f"Current weather in {CITY_NAME}: "
                      f"{weather_desc}. Temperature: {temperature}°C, "
                      f"Humidity: {humidity}%")
            return report
        else:
            return "City not found."
    except Exception as e:
        return "Failed to get weather information"

def search_google(query):
    """Open Google search results in browser"""
    search_url = f"https://google.com/search?q={query}"
    webbrowser.open(search_url)
    return f"Here are the Google results for {query}"

def search_wikipedia(query):
    """Search Wikipedia and return summary"""
    try:
        results = wikipedia.summary(query, sentences=2)
        return results
    except wikipedia.exceptions.DisambiguationError as e:
        return "Multiple results found. Please be more specific."
    except wikipedia.exceptions.PageError as e:
        return "No results found."

def process_command(command):
    """Process user commands"""
    if "weather" in command:
        weather_report = get_weather()
        speak(weather_report)
    elif "wikipedia" in command:
        speak("What would you like to search on Wikipedia?")
        search_query = listen()
        result = search_wikipedia(search_query)
        speak(result)
    elif "google" in command:
        speak("What would you like to search on Google?")
        search_query = listen()
        result = search_google(search_query)
        speak(result)
    else:
        speak("I'm sorry, I didn't understand that command.")

def wake_word_detected():
    """Check if wake word 'lisa' is detected"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Waiting for wake word...")
        audio = r.listen(source)
        
    try:
        detected = r.recognize_google(audio).lower()
        return "lisa" in detected
    except:
        return False

def main():
    speak("How can I assist you sir?")
    
    while True:
        if wake_word_detected():
            speak("How can I help you?")
            command = listen()
            
            if "exit" in command or "quit" in command:
                speak("Goodbye!")
                break
                
            process_command(command)

if __name__ == "__main__":
    # Install required packages if missing
    required_packages = [
        "SpeechRecognition",
        "pyttsx3",
        "wikipedia",
        "requests",
        "pyaudio"
    ]
    
    main()