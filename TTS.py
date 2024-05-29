import os
from deepgram import DeepgramClient, SpeakOptions
from dotenv import load_dotenv

load_dotenv()

def text_to_speech(transcript):
    os.environ["DEEPGRAM_API_KEY"] = os.environ.get("DEEPGRAM_API_KEY")

    try:
        deepgram = DeepgramClient()
        speak_options = {"text": transcript}

        options = SpeakOptions(
            model="aura-stella-en",
            encoding="linear16",
            container="wav"
        )

        response = deepgram.speak.v("1").stream(speak_options, options)

        return response.stream.getvalue()

    except Exception as e:
        print(f"Exception: {e}")