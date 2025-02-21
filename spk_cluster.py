import sys,os
import librosa,soundfile
import editdistance
import shutil
import glob
from tqdm import tqdm
import copy
import datetime
from modelscope.pipelines import pipeline
import torch

sv_pipeline_res = pipeline(
    task='speaker-verification',
    model='damo/speech_eres2net_sv_zh-cn_16k-common',
    model_revision='v1.0.5'
)
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
                res_list.append([basename + id, start, end, text, distance])
    return res_list

if __name__ == "__main__":
    raw_dir = "/home/jovyan/work/video-subtitle-extractor/data/raw_data/"
    if len(sys.argv) < 2:
        print("Need show name")
        sys.exit(0)
    show_name = sys.argv[1]
    dirpaths = sorted(glob.glob(raw_dir + "/*/"))
    res_list = list()
    for dirpath in dirpaths:
        text_path = dirpath + "/text"
        hyp_file = dirpath + "/test_label_checker.txt"
        final_list_path = dirpath + "/final_list.tsv"
        if os.path.exists(text_path) and os.path.exists(hyp_file) and os.path.exists(final_list_path):
            output_dict = label_error_detection(text_path,hyp_file)
            #print(output_dict)
            res_list.extend(final_process(dirpath, output_dict, final_list_path))


    supervised_list = res_list[:]


    visited_set = set()

    speaker_dict = dict()
    embeds_dict = dict()
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in range(len(supervised_list)-1):
        if not supervised_list[i][0] in visited_set and os.path.getsize(supervised_list[i][0]+".wav") <= 960000:
            visited_set.add(supervised_list[i][0])
            tmp_list = list()
            tmp_list.append(supervised_list[i])
            for j in range(i+1, len(supervised_list)):
                if not supervised_list[j][0] in visited_set and os.path.getsize(supervised_list[j][0]+".wav") <= 960000:
                    if supervised_list[i][0] in embeds_dict and supervised_list[j][0] in embeds_dict:
                        score = similarity(torch.from_numpy(embeds_dict[supervised_list[i][0]]), torch.from_numpy(embeds_dict[supervised_list[j][0]])).item()
                    else:
                        print(len(embeds_dict)/len(supervised_list))
                        result = sv_pipeline_res([supervised_list[i][0]+".wav" , supervised_list[j][0]+".wav"], output_emb=True)
                        score = result['outputs']['score']

                        embeds_dict[supervised_list[i][0]] = result['embs'][0]
                        embeds_dict[supervised_list[j][0]] = result['embs'][1]

                        #torch.cuda.empty_cache() 
                    if score >= 0.7: 
                        tmp_list.append(supervised_list[j])
                        visited_set.add(supervised_list[j][0])
            speaker_dict[supervised_list[i][0]] = tmp_list

    
    speaker_id = 1
    
    for i in speaker_dict:
        id = show_name + "_" + "SPK" + str("%04d" % speaker_id)
        if len(speaker_dict[i]) >= 5:
           
           print(id, len(speaker_dict[i]))
           #print(speaker_dict[i])
           speaker_id += 1
           save_dir = "/home/jovyan/work/video-subtitle-extractor/data/output_data/" + show_name + "/" + id
           os.makedirs(save_dir, exist_ok=True)
           for ii in speaker_dict[i]:
               vocal_wav = "/".join(ii[0].split("/")[:-1]) + "_vocals.wav"
               wav_id = id + "_" + "_".join(ii[0].split("/")[-2:])
               start = ii[1]
               end = ii[2]
               text = ii[3]
               text_path = save_dir + "/" + wav_id + ".txt"
               os.system(f"ffmpeg -loglevel quiet -ss {start} -to {end} -i '{vocal_wav}' -ar 48000 -ac 1 -acodec pcm_s16le {save_dir}/{wav_id}.wav")
               with open(text_path, "w") as f:
                   f.write(text + "\n")
               
            
        
