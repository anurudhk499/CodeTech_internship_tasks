#1.For Pre-recorded audio (compile time)
#You already have an audio file saved (like MP3 or WAV).
#Your program loads that file and transcribes it.
#Good for batch processing, files from phone, podcasts, etc."""
from pydub import AudioSegment
import speech_recognition as sr
import os

mp3_path = r"C:\Users\anuru\OneDrive\Attachments\Desktop\speeech_recognation\female.MP3"
wav_path = r"C:\Users\anuru\OneDrive\Attachments\Desktop\speeech_recognation\converted_audio.wav"

if not os.path.exists(mp3_path):
    print("❌ MP3 file NOT found! Please check the file path.")
    exit()

print("✅ MP3 file found! Starting conversion to WAV...")

audio = AudioSegment.from_mp3(mp3_path)
audio.export(wav_path, format="wav")

print(f"✅ Conversion done! WAV file saved at:\n{wav_path}")

recognizer = sr.Recognizer()
with sr.AudioFile(wav_path) as source:
    audio_data = recognizer.record(source)

try:
    print("📝 Transcribing audio...")
    text = recognizer.recognize_google(audio_data)
    print("\n🎙️ Transcription Result:\n", text)
except sr.UnknownValueError:
    print("❌ Could not understand the audio.")
except sr.RequestError as e:
    print(f"❌ Could not request results; {e}")

#2. Live voice input (runtime)
#The program uses your microphone to capture speech live.
#It listens in real-time or for a specified duration.
#Then sends that live audio data for transcription.
#Great for voice assistants, dictation apps, interactive programs.
import speech_recognition as sr

def live_speech_to_text():
    """
    Real-time Speech Recognition using the microphone
    Utilizes Google's Web Speech API (requires internet)
    """
    recognizer = sr.Recognizer()
    
    # Adjusting for ambient noise
    with sr.Microphone() as source:
        print("🎙️ Calibrating microphone for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Ready to listen(Say 'stop' to stop)\n")
        print("🎤 Listening... Speak now!")

        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                # 🔍 Recognize speech
                text = recognizer.recognize_google(audio).lower()
                print(f"🔊 You said: {text}\n")
                if text == "stop":
                    print("🛑 Stopping recognition...")
                    break    
            except sr.WaitTimeoutError:
                print("Listening timed out... Waiting for speech.")
            except sr.UnknownValueError:
                print("⚠️ I couldn't understand that. Please speak clearly.")
            except sr.RequestError as e:
                print(f"❌ API error: {e}")
                break

if __name__ == "__main__":
    live_speech_to_text()



