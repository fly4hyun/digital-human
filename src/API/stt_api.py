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

##########################################################################################

# Flask 서버 설정
app = Flask(__name__)

stt_results = Queue()

# STT 설정
# MODEL_TYPE = "medium"
# MODEL_TYPE = "large-v3"
MODEL_TYPE = "small"
LANGUAGE = "Korean"
#BLOCKSIZE = 24678
BLOCKSIZE = int(16000 * 1)
SILENCE_THRESHOLD = 3200
SILENCE_RATIO = 200
model = whisper.load_model(MODEL_TYPE)

global_ndarray = None

##########################################################################################

# 기존 STT 코드
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
    
    slience_check = 0
    print('Activating wire ...')
    async for indata, status in inputstream_generator():
        indata_flattened = abs(indata.flatten())

        # discard buffers that contain mostly silence
        ##### 입력 블록 중에 SILENCE_THRESHOLD 보다 큰 소리의 개수가 SILENCE_RATIO 보다 작으면 침묵으로 판단하여 무시
        if (np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):# and global_ndarray is None:
            if slience_check == 0:
                if global_ndarray is not None:
                    slience_check = 1
                continue
        elif slience_check == 1:
            slience_check = 0

        
        if global_ndarray is not None:
            global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
        else:
            global_ndarray = indata

        # concatenate buffers if the end of the current buffer is not silent
        ##### indata_flattened 끝에 n개 (디폴트 100) 의 평균이 SILENCE_THRESHOLD / 20 보다 작으면 침묵.. 문장 끝으로 판단
        if (np.average((indata_flattened[-200:-1])) > SILENCE_THRESHOLD / 10) or slience_check != 1:
            continue

        local_ndarray = global_ndarray.copy()
        global_ndarray = None
        slience_check = 0
        indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
        result = model.transcribe(indata_transformed, language=LANGUAGE)
        
        stt_results.put(result["text"])  # 결과를 큐에 저장

        del local_ndarray
        del indata_flattened

        return result["text"]  # 결과를 바로 반환

##########################################################################################

@app.route('/detect-and-transcribe', methods=['POST'])
def detect_and_transcribe():

    # 비동기적으로 STT 처리를 시작
    asyncio.run(process_audio_buffer())
    
    # STT 결과
    result = stt_results.get() if not stt_results.empty() else 'Waiting for STT result'
    return jsonify({'text': result}), 200

    
async def main():
    audio_task = asyncio.create_task(process_audio_buffer())
    
def run_stt_event_loop():
    asyncio.run(main())

# Flask 앱을 실행하는 스레드
def run_flask_app():
    app.run(debug=False, port=5002)


# Flask 앱을 실행하는 스레드
        
##########################################################################################

if __name__ == '__main__':
    
    # Flask 앱을 실행하는 스레드 시작
    flask_thread = Thread(target=run_flask_app)
    flask_thread.start()

