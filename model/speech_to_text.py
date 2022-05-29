import speech_recognition as sr
from io import BytesIO
from scipy.io.wavfile import write, read
import base64

class Speech2Text(object):
    def __init__(self):
        self.r = sr.Recognizer()
    def convert2audio(self, file):
        rate, data = read(BytesIO(base64.b64decode(file)))
        write("speech/request.wav", rate, data)
        with sr.AudioFile("speech/request.wav") as source:
            audio = self.r.record(source)
        return audio
    def recognition(self, file):
        audio = self.convert2audio(file)
        try:
            s = self.r.recognize_google(audio, language="vi-VI")
            return s
        except Exception as e:
            return None


if __name__ == "__main__":
    with open("ques_1.wav", 'rb') as fd:
        contents = fd.read()
        # print(len(contents))

    rate, data = read(BytesIO(contents))
    write("a.wav", rate, data)
    print(rate)

    r = sr.Recognizer()
    with sr.AudioFile("a.wav") as source:
        audio = r.record(source)

    # try:
    s = r.recognize_google(audio_data=audio, language='vi-VI')
    print(s)
    # except Exception as e:
    #     print("Error")

