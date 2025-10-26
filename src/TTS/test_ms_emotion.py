

###################################################################################################

import IPython.display as ipd
import torch
from torch.utils.data import DataLoader

import numpy as np
from scipy.io.wavfile import write
import math
import time

from utils.task import load_checkpoint
from utils.hparams import get_hparams_from_file
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from model.models import SynthesizerTrn
#from text.symbols import symbols
from text import tokenizer

from utils.task import load_vocab

###################################################################################################

start_all = time.time()

###################################################################################################

#ko_test = '별 하나에 추억과 별 하나에 사랑과 별 하나에 쓸쓸함과 별 하나에 동경과 별 하나에 시와 별 하나에 어머니, 어머니,'
#ko_test = '가슴 속에 하나 둘 새겨지는 별을 이제 다 못 헤는 것은 쉬이 아침이 오는 까닭이요 내일 밤이 남은 까닭이요 아직 나의 청춘이 다하지 않은 까닭입니다.'
#ko_test = '현대인의 하루는 플라스틱과 함께 시작된다. 플라스틱 튜브에 들어있는 치약과 플라스틱 칫솔을 사용하고, 안쪽이 플라스틱으로 코팅된 종이컵에 플라스틱으로 만들어진 정수기의 물과 필름지의 커피를 담아 습관처럼 하루에도 물과 커피를 마신다. 각종 플라스틱 부품이 들어간 컴퓨터와 플라스틱으로 만들어진 사무용 기기를 사용하고, 페트병에 든 물과 다양한 모양의 플라스틱 포장지에 든 음식을 먹는다. 단 한순간도 플라스틱 없는 세상을 상상할 수 없을 정도로 플라스틱 분수속에 살고 있는 현대인! 그러나 많이 쓰고 쉽게 접하는 만큼 많이 버려지는 것 또한 플라스틱이다. 대기 수질오염과 함께 제3의 산업공해라는 불명예를 안으며 대표적인 환경오염 물질로 떠오른 플라스틱. 환경을 생각하는 저탄소 녹색사회를 위해 과연 플라스틱의 미래는 어디로 가야하는 것일까?'
#ko_test = '이 몸이 죽고 죽어 일 백번 고쳐 죽어, 백골이 진토 되어 넋이라도 있고 없고, 임 향한 일편단심이야 가실 줄이 있으랴'
#ko_test = '서울의 번화한 명동에서 출발하여 서쪽으로 강을 따라 산책하면서 수많은 사람들의 일상과 그들의 이야기를 지켜보며, 도시의 흐름 속에서도 아직 남아있는 고즈넉한 한옥 마을, 전통적인 문화와 현대적인 스타일이 공존하는 모습, 그리고 그 사이에서 움켜잡힌 작은 이야기들을 발견할 수 있었다. 이 도시는 고대의 역사와 현대 도시의 모습이 조화를 이루면서, 여러 세대의 사람들이 그들만의 방식으로 삶을 즐기고 도전하며, 서로의 꿈을 키워나가는 공간으로 변모하였다.'
#ko_test = '오늘은 날씨가 참 좋네요! 항상 비올때마다 그렇게 생각하곤 했어요. 그럼에도 불구하고 비는 공기를 맑게 해주니까 좋아요. 당신은 어떤 날씨를 좋아하나요?'
#ko_test = '별빛이 아름다운 밤에 내 마음 아픈 일 생각하며 새별 빛나는 꿈과 사랑과 행복 찾아 별들을 셉니다. 꿈들은 마다 가고 무엇 하나 머문 것 없이 다 가버린다는 걸 알면서도 별빛은 언제나 내게 와 꿈과 사랑과 행복을 주려하나 봅니다.'
#ko_test = '계절이 지나가는 하늘에는 가을로 가득 차 있습니다. 나는 아무 걱정도 없이 가을 속의 별들을 다 헤일 듯합니다. 가슴속에 하나둘 새겨지는 별을 이제 다 못 헤는 것은 쉬이 아침이 오는 까닭이요, 내일 밤이 남은 까닭이요, 아직 나의 청춘이 다하지 않은 까닭입니다. 별 하나에 추억과 별 하나에 사랑과 별 하나에 쓸쓸함과 별 하나에 동경과 별 하나에 시와 별 하나에 어머니, 어머니, 어머님, 나는 별 하나에 아름다운 말 한마디씩 불러 봅니다. 소학교 때 책상을 같이 했던 아이들의 이름과, 패, 경, 옥, 이런 이국 소녀들의 이름과, 벌써 아기 어머니 된 계집애들의 이름과, 가난한 이웃 사람들의 이름과, 비둘기, 강아지, 토끼, 노새, 노루, 프랑시스 잠, 라이너 마리아 릴케 이런 시인의 이름을 불러 봅니다. 이네들은 너무나 멀리 있습니다. 별이 아스라이 멀듯이. 어머님, 그리고 당신은 멀리 북간도에 계십니다. 나는 무엇인지 그리워 이 많은 별빛이 내린 언덕 위에 내 이름자를 써 보고 흙으로 덮어 버리었습니다. 딴은 밤을 새워 우는 벌레는 부끄러운 이름을 슬퍼하는 까닭입니다. 그러나 겨울이 지나고 나의 별에도 봄이 오면 무덤 위에 파란 잔디가 피어나듯이 내 이름자 묻힌 언덕 위에도 자랑처럼 풀이 무성할 거외다.'
#ko_test = '계절이 지나가는 하늘에는 가을로 가득 차 있습니다. 나는 아무 걱정도 없이 가을 속의 별들을 다 헤일 듯합니다. 가슴속에 하나둘 새겨지는 별을 이제 다 못 헤는 것은 쉬이 아침이 오는 까닭이요 내일 밤이 남은 까닭이요 아직 나의 청춘이 다하지 않은 까닭입니다. 별 하나에 추억과 별 하나에 사랑과 별 하나에 쓸쓸함과 별 하나에 동경과 별 하나에 시와 별 하나에 어머니 어머니 어머님 나는 별 하나에 아름다운 말 한마디씩 불러 봅니다. 소학교 때 책상을 같이 했던 아이들의 이름과 패 경 옥 이런 이국 소녀들의 이름과 벌써 아기 어머니 된 계집애들의 이름과 가난한 이웃 사람들의 이름과 비둘기 강아지 토끼 노새 노루 프랑시스 잠 라이너 마리아 릴케 이런 시인의 이름을 불러 봅니다. 이네들은 너무나 멀리 있습니다. 별이 아스라이 멀듯이. 어머님 그리고 당신은 멀리 북간도에 계십니다. 나는 무엇인지 그리워 이 많은 별빛이 내린 언덕 위에 내 이름자를 써 보고 흙으로 덮어 버리었습니다. 딴은 밤을 새워 우는 벌레는 부끄러운 이름을 슬퍼하는 까닭입니다. 그러나 겨울이 지나고 나의 별에도 봄이 오면 무덤 위에 파란 잔디가 피어나듯이 내 이름자 묻힌 언덕 위에도 자랑처럼 풀이 무성할 거외다.'

ko_test = '현대인의 하루는 플라스틱과 함께 시작된다.'

id_num = 0
emo = 'sur'

model = "emotion"
checkpoint = "G_84000.pth"

# G_449000.pth : 10000 (9978)
# G_225000.pth : 5000 (5001)

len_symbols = 133

###################################################################################################

emo_mapping = {
    'ang': '5', 
    'dis': '6', 
    'fea': '7',
    'hap': '8', 
    'neu': '11', 
    'sad': '9', 
    'sur': '10', 
}

###################################################################################################

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

###################################################################################################

def korean_to_embeding(word):
    
    word_list = []
    
    if '가' <= word <= '힣':
        ch1 = (ord(word) - ord('가'))//588
        ch2 = ((ord(word) - ord('가')) - (588*ch1)) // 28
        ch3 = (ord(word) - ord('가')) - (588*ch1) - 28*ch2
        
        word_list.append(CHOSUNG_LIST[ch1])
        word_list.append(JUNGSUNG_LIST[ch2])
        if JONGSUNG_LIST[ch3] != ' ':
            word_list.append(JONGSUNG_LIST[ch3])
    
    elif word == ' ':
        word_list.append('<space>')
    
    else:
        word_list.append(word)
        
    return word_list
        
###################################################################################################

with open("./datasets/emotion/vocab.txt", "r") as f:
    
    vocab_lines = f.readlines()

word_mapping = {}

for v_line in vocab_lines:
    
    w_one, w_index = v_line.split('\t')
    w_index = w_index[:-1]
    
    word_mapping[w_one] = w_index

###################################################################################################




###################################################################################################

def get_text(text: str, emo, hps) -> torch.LongTensor:
    vocab = load_vocab(hps.data.vocab_file)
    cleaned_text = getattr(hps.data, "cleaned_text", False)
    
    
    text_emb = ['2']
    for t in text:
        t_embs = korean_to_embeding(t)
        
        for t_emb in t_embs:

            if t_emb in list(word_mapping.keys()):
                text_emb += [word_mapping[t_emb]]
            else:
                text_emb += ['1']

            
    text_emb += ['3']
    
    text_emb = '\t'.join(text_emb)

    text_emb = emo_mapping[emo] + '\t' + text_emb + '\t' + emo_mapping[emo]
        
    text_norm = tokenizer(text_emb, vocab, hps.data.text_cleaners, language=hps.data.language, cleaned_text=cleaned_text)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

###################################################################################################



###################################################################################################






hps = get_hparams_from_file(f"./datasets/{model}/config.yaml")
filter_length = hps.data.n_mels if hps.data.use_mel else hps.data.n_fft // 2 + 1
segment_size = hps.train.segment_size // hps.data.hop_length
#net_g = SynthesizerTrn(len(symbols), filter_length, segment_size, **hps.model).cuda()
net_g = SynthesizerTrn(len_symbols, filter_length, segment_size, n_speakers=hps.data.n_speakers, **hps.model).cuda()
_ = net_g.eval()
_ = load_checkpoint(f"./datasets/{model}/logs/{checkpoint}", net_g, None)

###################################################################################################






start = time.time()





###################################################################################################


stn_tst = get_text(ko_test, emo, hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([id_num]).cuda()

    out = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)
    audio = out[0][0, 0].data.cpu().float().numpy()
    
    scaled_data = np.int16(audio/np.max(np.abs(audio)) * 32767)

    write("emotion_results/output_vits2_emotion_" + str(emo) + ".wav", hps.data.sample_rate, scaled_data)
    write("emotion_results/output_vits2_emotion.wav", hps.data.sample_rate, scaled_data)

end = time.time()
end_all = time.time()


print(f"{end - start:.5f} sec")
print(f"{end_all - start_all:.5f} sec")


