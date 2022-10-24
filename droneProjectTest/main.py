import speech_recognition as sr

Recognizer = sr.Recognizer()  # 인스턴스 생성
mic = sr.Microphone()

while True:
    inputText = input("음성 인식을 위해 'start' 를 입력하세요 : ");

    if inputText == 'start':
        with mic as source:  # 안녕~이라고 말하면
            audio = Recognizer.listen(source)
        try:
            data = Recognizer.recognize_google(audio, language="ko")
        except:
            print("이해하지 못했음")

        print(data)  # 안녕 출력

    elif inputText == 'end':
        exit()
    else:
        continue
