import sys,os
import librosa,soundfile
import editdistance
import shutil
import glob
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import copy
import datetime
from modelscope.pipelines import pipeline

sv_pipeline_res = pipeline(
    task='speaker-verification',
    model='damo/speech_eres2net_sv_zh-cn_16k-common',
    model_revision='v1.0.5'
)


def timestamp_to_milliseconds(timestamp):
    """
    接受一个'00:00:03,567 --> 00:00:05,866'类型的时间戳，将其转化为毫秒（87.234）
    """
    h, m, s, ms = timestamp.split('-->')[0].strip().replace(',', ':').split(':')
    
    total_s = float(int(m) * 60 + int(h) * 60 * 60 + int(s)) + int(ms)*0.001

    h, m, s, ms = timestamp.split('-->')[1].strip().replace(',', ':').split(':')

    total_e = float(int(m) * 60 + int(h) * 60 * 60 + int(s)) + int(ms)*0.001
    return [total_s, total_e]

def load_srt_file(srt_file):
    output_list = list()
    fo = open(srt_file, "r+", encoding='utf-8')

    while (True):
        try:
            id = fo.readline().strip()
            timestamp = fo.readline().strip()
            text = fo.readline().strip().replace(" ","")
            blank = fo.readline().strip()
            while len(blank) > 0:
                blank = fo.readline().strip() #protect multiple lines
        except Exception as e:
            print('nothing of except:', e)
            break
        if (id == "" or timestamp == "" or text == ""):
            break
        else:
            tmp_list = []
            tmp_list.append(int(id))
            tmp_list.extend(timestamp_to_milliseconds(timestamp))
            tmp_list.append(text)
            output_list.append(tmp_list)
            
    return output_list

def merge_timeline_v1(time_list):
    output_list = list()
    for part in time_list:
        if len(output_list) > 0:
            if part[-1] == output_list[-1][-1]:
                output_list[-1][2] = part[2]
            else:
                if part[2] - part[1] > 0.1:
                    output_list.append(part[:])
        else:
            if part[2] - part[1] > 0.1:
                output_list.append(part[:])
        
    return output_list

def speaker_merge(save_dir, tmp16_file, merged_list):
    output_list = list()
    tmpsv_file = save_dir + "/tmp_sv.wav"
    tmpold_file = save_dir + "/tmp_old.wav"

    if os.path.exists(tmpsv_file):
        os.remove(tmpsv_file)

    if os.path.exists(tmpold_file):
        os.remove(tmpold_file)
        
    for part in merged_list:
        if len(output_list) > 0:
            os.system(f"ffmpeg -loglevel quiet -ss {part[1]} -to {part[2]} -i '{tmp16_file}' -ar 16000 -ac 1 -acodec pcm_s16le {tmpsv_file}")
            if part[1] - output_list[-1][2] < 5.0 and part[2] - part[1] < 20.0 and output_list[-1][2] - output_list[-1][1] < 20.0:
                result = sv_pipeline_res([tmpold_file,tmpsv_file])
            
                if result['score'] >= 0.5:
                    output_list[-1][2] = part[2]
                    output_list[-1][3] += part[3]
                else:
                    output_list.append(part[:])
            else:
                output_list.append(part[:])
            os.system(f"mv {tmpsv_file} {tmpold_file}")
                
        else:
            output_list.append(part[:])
            os.system(f"ffmpeg -loglevel quiet -ss {part[1]} -to {part[2]} -i '{tmp16_file}' -ar 16000 -ac 1 -acodec pcm_s16le {tmpold_file}")
        #print(output_list[-1])
            
    return output_list

def merge_timeline_v2(merged_list):
    threshold = 0.05
    output_list = list()

    for part in merged_list:
        tmp_list = part[:]
        if len(output_list) > 0:
            if tmp_list[1] - output_list[-1][2] >= 2.0:
                output_list[-1][2] += 1.0
                output_list.append(tmp_list)
                output_list[-1][1] -= 1.0
            else:
                output_list.append(tmp_list)
                end_point = output_list[-2][2]
                start_point = output_list[-1][1]
                output_list[-2][2] = 0.8 *end_point + 0.2*start_point
                output_list[-1][1] = output_list[-2][2]
        else:
            output_list.append(tmp_list)
            if tmp_list[1] > 1.0:
                output_list[-1][1] -= 1.0
           
    return output_list

def save_part(tmp16_file, merged_list, save_dir):
    fwscp = open(save_dir + "/wav.scp","w")
    ftext = open(save_dir + "/text","w")
    flist = open(save_dir + "/final_list.tsv","w")
    for clip in merged_list:
        id, start, end, text = clip
        id = str("%08d" % id)
        os.system(f"ffmpeg -loglevel quiet -ss {start} -to {end} -i '{tmp16_file}' -ar 16000 -ac 1 -acodec pcm_s16le {save_dir}/{id}.wav")
        with open(os.path.join(save_dir + "/", f'{id}.txt'),"w") as f:
            f.write(text + "\n")
        ftext.write(str(id) + " " + text + "\n")
        fwscp.write(str(id) + " " + os.path.join(os.path.abspath(save_dir)  + "/" + f'{id}.wav') + "\n")
        flist.write(str(id) + "\t" + str(start) + "\t" + str(end) + "\n")
    fwscp.close()
    ftext.close()
    flist.close()
    return os.path.abspath(save_dir) + "/text", os.path.abspath(save_dir) + "/wav.scp", os.path.abspath(save_dir) + "/final_list.tsv"

def _label_check_main(text_path, wav_scp_path, res_dir):
    os.system(f"/home/jovyan/work/wenet/runtime/libtorch/build/bin/label_checker_main \
                --model_path /home/jovyan/work/wenet/runtime/libtorch/build/bin/20220506_u2pp_conformer_libtorch//final.zip \
                --unit_path /home/jovyan/work/wenet/runtime/libtorch/build/bin/20220506_u2pp_conformer_libtorch//units.txt \
                --text {text_path} \
                --wav_scp {wav_scp_path} \
                --result {res_dir}/test_label_checker.txt \
                --timestamp {res_dir}/timestamp.txt \
                --del_penalty 2.3 \
                --is_penalty 9.2 ")
    return f"{res_dir}/test_label_checker.txt"

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
        
def load_text_file(file_path):
    output_dict = dict()
    with open(file_path, "r") as f:
        for line in f:
            line = line.rstrip()
            datas = line.split(" ")
            flag = True
            if len(datas) == 2 :
                text = datas[1].replace("<del>","").replace("<is>","").replace("</is>","")
                for character in text:
                    flag = flag & is_chinese(character)
                if flag:
                    output_dict[datas[0]] = text
    return output_dict

def label_error_detection(ref_file, hyp_file):
    ref_dict = load_text_file(ref_file)
    hyp_dict = load_text_file(hyp_file)

    output_dict = dict()
    for key in ref_dict:
        ref = ref_dict[key]
        if key in hyp_dict:
            hyp = hyp_dict[key]
            c = 1.0 - 1.0 * editdistance.eval(ref, hyp) / max(len(ref), len(hyp))
            if c > 0.95:
                output_dict[key] = [hyp, c]
                #print(key, hyp, str(c))
            else:
                output_dict[key] = [ref, c]
                #print(key, ref, str(c))
    return output_dict

def final_process(basename, output_dict, final_list_path):
    res_list = list()
    final_list = list()
    with open(final_list_path, "r") as f:
        for line in f:
            id, start, end = line.rstrip().split("\t")
            final_list.append([id, start, end])
    for part in final_list:
        id, start, end = part
        if id in output_dict:
            text = output_dict[id][0]
            distance = output_dict[id][1]
            if distance > 0.95:
                res_list.append([basename + "/" + id, start, end, text, distance])
    return res_list

def label_error_process(dirpath):
    text_path = dirpath + "/text"
    wav_scp_path = dirpath + "/wav.scp"
    final_list_path = dirpath + "/final_list.tsv"
    tmp = os.path.abspath(dirpath)
    hyp_file = f"{tmp}/test_label_checker.txt"
    hyp_file = _label_check_main(text_path, wav_scp_path, os.path.abspath(dirpath))


    

if __name__ == "__main__":
    raw_dir = "/home/jovyan/work/video-subtitle-extractor/data/raw_data/"
    srt_files = sorted(glob.glob(raw_dir + "/" + "*.srt"))
    for srtfile in srt_files:
        vocal_wavfile = srtfile.replace(".srt","_vocals.wav")
        if os.path.exists(vocal_wavfile):
            save_dir = srtfile.replace(".srt","")
            os.makedirs(save_dir, exist_ok=True)
            srt_list = load_srt_file(srtfile)
            print(srt_list)
            print(srtfile,"Done load srt")
            merged_list = merge_timeline_v1(srt_list)
            print(merged_list)
            print(srtfile,"Done merge srt")
            speaker_merged_list = speaker_merge(save_dir, vocal_wavfile, merged_list)
            print(speaker_merged_list)
            print(srtfile,"Done merge speaker")
            final_list = merge_timeline_v2(speaker_merged_list)
            print(final_list)
            text_path, wav_scp_path, time_scp_path = save_part(vocal_wavfile, final_list, save_dir)
            print(srtfile,"Done save part")

    dirpaths = sorted(glob.glob(raw_dir + "/*/"))
    thread_num = len(dirpaths)
    pool = Pool(thread_num)
    pool.map(label_error_process, dirpaths)
    pool.close()

    
