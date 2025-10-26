import sounddevice as sd
import numpy as np
import whisper
import asyncio
import queue
import sys

# # SETTINGS
# MODEL_TYPE = "medium"
# # the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
# LANGUAGE = "Korean"
# # pre-set the language to avoid autodetection
# BLOCKSIZE = 24678
# # this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds.
# SILENCE_THRESHOLD = 400
# # should be set to the lowest sample amplitude that the speech in the audio material has
# SILENCE_RATIO = 200
# # number of samples in one buffer that are allowed to be higher than threshold

# MODEL_TYPE = "small"
MODEL_TYPE = "medium"
LANGUAGE = "Korean"
BLOCKSIZE = int(16000 * 1)
SILENCE_THRESHOLD = 2000
SILENCE_RATIO = 200

def read_and_concatenate_words(file_path):
    # 파일에서 단어들을 읽음
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()

    # 모든 단어들을 공백 하나로 이어붙임
    concatenated_words = ' '.join(words)

    return concatenated_words
noun_list = read_and_concatenate_words('Noun_list.txt')

global_ndarray = None
model = whisper.load_model(MODEL_TYPE)

print("model")
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


async def process_audio_buffer():
    global global_ndarray
    
    slience_check = 0
    
    async for indata, status in inputstream_generator():
        
        indata_flattened = abs(indata.flatten())
        print('start1')
        # discard buffers that contain mostly silence
        ##### 입력 블록 중에 SILENCE_THRESHOLD 보다 큰 소리의 개수가 SILENCE_RATIO 보다 작으면 침묵으로 판단하여 무시
        if (np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):# and global_ndarray is None:
            if slience_check == 0:
                if global_ndarray is not None:
                    slience_check = 1
                continue
        elif slience_check == 1:
            slience_check = 0

        print('start2')
        if global_ndarray is not None:
            global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
        else:
            global_ndarray = indata
        
        
        
        # concatenate buffers if the end of the current buffer is not silent
        ##### indata_flattened 끝에 n개 (디폴트 100) 의 평균이 SILENCE_THRESHOLD / 20 보다 작으면 침묵.. 문장 끝으로 판단
        
        if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD / 5) or slience_check != 1:
            continue
        
        else:
            
            local_ndarray = global_ndarray.copy()
            
            global_ndarray = None
            slience_check = 0
            
            indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            
            result = model.transcribe(indata_transformed, language=LANGUAGE, initial_prompt=noun_list)
            print(result["text"])
            
            #return result["text"]  # 결과를 바로 반환합니다.

        del local_ndarray
        del indata_flattened
        


async def main():
    print('Activating wire ...')
    audio_task = asyncio.create_task(process_audio_buffer())
    while True:
        await asyncio.sleep(1)
    audio_task.cancel()
    try:
        await audio_task
    except asyncio.CancelledError:
        print('wire was cancelled')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('Interrupted by user')
