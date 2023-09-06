import os

# 1、使用moviepy模块 提取视频中的音频文件
from moviepy.editor import AudioFileClip
import time
from faster_whisper import WhisperModel
import math

#输入视频，生成视频的字幕，支持英语中文，目前使用中文

all_start = time.time()

videofile = "D:/000000/pycharm project/file/1693562345281.mp4"                     #视频文件
modelckt = "D:/000000/pycharm project/ckt/faster-whisper-large-v2"           #音频转文字模型文件
chatglmmodelckt = "D:\\000000\\pycharm project\\ckt\\chatglm2-6b-32k-int4"   #chatglm2的模型文件

videorootpath = '/'.join(videofile.split('/')[0:-1])
videofilename = videofile.split('/')[-1].split('.')[0]
tempaudiofile = videorootpath + '/' + videofilename + '.mp3'      #生成的临时音频文件
savesrtfile = videorootpath + '/' + videofilename + '.srt'       #生成的字幕文件

print("开始从视频中提取音频...")
if os.path.exists(savesrtfile) == False:
    my_audio_clip = AudioFileClip(videofile)
    my_audio_clip.write_audiofile(tempaudiofile)
else:
    print("音频文件已存在")

print('开始音频转文字...')
T1 = time.time()
model = WhisperModel(modelckt, device="cuda", compute_type="float16")  # Silero VAD模型过滤掉没有语音的音频部分
segments, info = model.transcribe(tempaudiofile, beam_size=5,language='zh', vad_filter=True)
T2 = time.time()
# os.remove(tempaudiofile)         #可以删除
print('识别运行时间:%s秒!' % (T2 - T1))      # 时间：small+audio.mp3 = 9.5s。small+1693562345281.mp3=101s

#将秒计时的时间转换成lrc时间格式
def sec2time(start):
    start_ms,a = math.modf(start)
    start_ms = start_ms*1000
    start_h = int(a/3600)
    start_m = int((a%3600)/60)
    start_s = int((a%3600)%60)
    start_str = "%02d:%02d:%02d.%03d" % (start_h, start_m, start_s,start_ms)
    return start_str

print("保存音频识别结果...")
result = []
t1 = time.time()
n = 0
if os.path.exists(savesrtfile):   #如果生成过一次，就删除掉，确保每一次都是从头生成
    os.remove(savesrtfile)
for segment in segments:
    start_str = sec2time(segment.start)
    end_str = sec2time(segment.end)
    lrcwrite = str(n) + '\n' + start_str + ' --> ' + end_str + '\n' + segment.text + '\n\n'
    with open(savesrtfile, 'a+', encoding="utf-8") as f:
        f.write(lrcwrite)
    n = n + 1
t2 = time.time()
print("字幕保存完毕！")
print('文本转字幕时间:%s秒!' % (t2-t1))
print('程序总运行时间:%s秒!' % (time.time()-all_start))
