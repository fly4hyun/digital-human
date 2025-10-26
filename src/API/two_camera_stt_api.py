from flask import Flask, jsonify
import threading
from threading import Thread
import requests
import sounddevice as sd
import numpy as np
import time
import whisper
import asyncio
from queue import Queue

# Flask 서버 설정
app = Flask(__name__)

# 사람 감지 상태를 저장할 큐
detection_queue = Queue()
stt_results = Queue()

# STT 설정
MODEL_TYPE = "medium"
# MODEL_TYPE = "large-v3"
# MODEL_TYPE = "small"
LANGUAGE = "Korean"
#BLOCKSIZE = 24678
BLOCKSIZE = 16000 * 3
SILENCE_THRESHOLD = 400
SILENCE_RATIO = 200
model = whisper.load_model(MODEL_TYPE)

global_ndarray = None
detection_status = None

delay_time = 1




# 여기부터는 기존 STT 코드를 포함시킵니다.
async def inputstream_generator():
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status

# STT 결과를 저장할 변수
stt_results = Queue()
    
    
async def process_audio_buffer():
    global global_ndarray
    async for indata, status in inputstream_generator():
        indata_flattened = abs(indata.flatten())
        print('start1')
        print(len(indata_flattened))
        
        # discard buffers that contain mostly silence
        ##### 입력 블록 중에 SILENCE_THRESHOLD 보다 큰 소리의 개수가 SILENCE_RATIO 보다 작으면 침묵으로 판단하여 무시
        if (np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):
            continue

        print('start2')
        if global_ndarray is not None:
            global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
        else:
            global_ndarray = indata
        print('start3')
        
        
        # concatenate buffers if the end of the current buffer is not silent
        ##### indata_flattened 끝에 n개 (디폴트 100) 의 평균이 SILENCE_THRESHOLD / 20 보다 작으면 침묵.. 문장 끝으로 판단
        if (np.average((indata_flattened[-16000:-1])) > SILENCE_THRESHOLD / 20):
            continue
        print(1)
        local_ndarray = global_ndarray.copy()
        print(2)
        global_ndarray = None
        print(3)
        indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
        print(4)
        result = model.transcribe(indata_transformed, language=LANGUAGE)
        print(result["text"])
        stt_results.put(result["text"])  # 결과를 큐에 저장
        print('start4')
        return result["text"]  # 결과를 바로 반환

@app.route('/detect-and-transcribe', methods=['POST'])
def detect_and_transcribe():
    
    global detection_status

    detection_status = detection_queue.get() if not detection_queue.empty() else '0'
    if detection_status == '1':
        # 비동기적으로 STT 처리를 시작
        asyncio.run(process_audio_buffer())
        
        detection_status = '0'  # 처리가 완료되면 상태를 재설정
        # STT 결과를 가져옵니다.
        result = stt_results.get() if not stt_results.empty() else 'Waiting for STT result'
        return jsonify({'text': result}), 200
    else:
        return jsonify({'message': 'No human detected'}), 200
    
async def main():
    print('Activating wire ...')
    audio_task = asyncio.create_task(process_audio_buffer())
    
def run_stt_event_loop():
    asyncio.run(main())

# Flask 앱을 실행하는 스레드
def run_flask_app():
    app.run(debug=False, port=5001)


# Flask 앱을 실행하는 스레드


################

# 사람 탐지 상태를 주기적으로 가져오는 스레드 실행
def detection_status_updater():
    global detection_status  # 전역 변수로 사용하겠다고 선언
    while True:
        detect_api_url = 'http://localhost:5000/detect-human'
        response = requests.post(detect_api_url)
        if response.status_code == 200:
            current_status = response.json().get('human_detected', '0')
            elapsed_time = response.json().get('elapsed_time', 1)
            if current_status == '1' and detection_status == '0':
                detection_queue.put(current_status)
                detection_status = '1'  # 전역 변수를 업데이트
        sleep_time = max(delay_time - elapsed_time, 0)
        time.sleep(sleep_time)
        detection_status = '0'  # 전역 변수를 업데이트
        
        
if __name__ == '__main__':
    # STT 처리를 관리할 이벤트 루프를 실행하는 스레드 시작


    # 사람 탐지 상태를 주기적으로 가져오는 스레드 실행
    detection_thread = Thread(target=detection_status_updater, daemon=True)
    detection_thread.start()
    
    # Flask 앱을 실행하는 스레드 시작
    flask_thread = Thread(target=run_flask_app)
    flask_thread.start()

