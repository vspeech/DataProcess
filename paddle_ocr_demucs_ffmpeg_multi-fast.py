import sys,os
import librosa,soundfile
import editdistance
import shutil
import glob
from tqdm import tqdm
from functools import partial
#from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import copy
import datetime

import cv2
from skimage.metrics import structural_similarity
import srt
import fastdeploy as fd
import numpy as np

def load_model(runtime_option):
    # Detection模型, 检测文字框
    det_model_file = os.path.join("./ch_PP-OCRv3_det_infer", "inference.pdmodel")
    det_params_file = os.path.join("./ch_PP-OCRv3_det_infer", "inference.pdiparams")
    # Recognition模型，文字识别模型
    rec_model_file = os.path.join("./ch_PP-OCRv3_rec_infer", "inference.pdmodel")
    rec_params_file = os.path.join("./ch_PP-OCRv3_rec_infer", "inference.pdiparams")
    rec_label_file = "ppocr_keys_v1.txt"

    # PPOCR的cls和rec模型现在已经支持推理一个Batch的数据
    # 定义下面两个变量后, 可用于设置trt输入shape, 并在PPOCR模型初始化后, 完成Batch推理设置
    rec_batch_size = -1

    # 当使用TRT时，分别给三个模型的runtime设置动态shape,并完成模型的创建.
    # 注意: 需要在检测模型创建完成后，再设置分类模型的动态输入并创建分类模型, 识别模型同理.
    # 如果用户想要自己改动检测模型的输入shape, 我们建议用户把检测模型的长和高设置为32的倍数.
    det_option = runtime_option
    #det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
    #                               [1, 3, 960, 960])
    # 用户可以把TRT引擎文件保存至本地
    #det_option.set_trt_cache_file("./ch_PP-OCRv3_det_infer" + "/det_trt_cache.trt")
    global det_model
    det_model = fd.vision.ocr.DBDetector(
        det_model_file, det_params_file, runtime_option=det_option)


    rec_option = runtime_option
    #rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
    #                               [rec_batch_size, 3, 48, 320],
    #                               [rec_batch_size, 3, 48, 2304])
    # 用户可以把TRT引擎文件保存至本地
    #rec_option.set_trt_cache_file("./ch_PP-OCRv3_rec_infer"  + "/rec_trt_cache.trt")
    global rec_model
    rec_model = fd.vision.ocr.Recognizer(
        rec_model_file,
        rec_params_file,
        rec_label_file,
        runtime_option=rec_option)

    # 创建PP-OCR，串联3个模型，其中cls_model可选，如无需求，可设置为None
    global ppocr_v3
    ppocr_v3 = fd.vision.ocr.PPOCRv3(
        det_model=det_model, cls_model=None, rec_model=rec_model)

    # 给cls和rec模型设置推理时的batch size
    # 此值能为-1, 和1到正无穷
    # 当此值为-1时, cls和rec模型的batch size将默认和det模型检测出的框的数量相同

    ppocr_v3.rec_batch_size = rec_batch_size

def box2int(box):
    for i in range(len(box)):
        for j in range(len(box[i])):
            box[i][j] = int(box[i][j])
    return box
    
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def detect_subtitle_area(ocr_results, h, w):
    '''
    Args:
        w(int): width of the input video
        h(int): height of the input video
    '''
    #ocr_results = ocr_results[0]  # 0, the first image result
    # Merge horizon text areas
    # Merge horizon text areas
    idx = 0
    candidates = []
    while idx < len(ocr_results):
        boxes, text = ocr_results[idx]
        idx += 1
        con_boxes = copy.deepcopy(boxes)
        con_text = text
        while idx < len(ocr_results):
            n_boxes, n_text = ocr_results[idx]
            if abs(n_boxes[0][1] - boxes[0][1]) < h * 0.04 and \
               abs(n_boxes[3][1] - boxes[3][1]) < h * 0.04:
                con_boxes[1] = n_boxes[1]
                con_boxes[2] = n_boxes[2]
                con_text = con_text + n_text
                idx += 1
            else:
                break
        candidates.append((con_boxes, con_text))
    # TODO(Binbin Zhang): Only support horion center subtitle
    if len(candidates) > 0:
        sub_boxes, subtitle = candidates[-1]
        subtitile = subtitle.replace(" ","")
        # offset is less than 10%
        #(sub_boxes[0][0] + sub_boxes[1][0]) / w > 0.90 and
        if len(subtitle) > 0 and abs(sub_boxes[1][0] + sub_boxes[0][0])*0.5 >= 0.4*w and abs(sub_boxes[1][0] + sub_boxes[0][0])*0.5 <= 0.6*w and abs(sub_boxes[1][0] - sub_boxes[0][0]) >= abs(sub_boxes[2][1] - sub_boxes[1][1]) and abs(sub_boxes[1][0] - sub_boxes[0][0]) > 7 and abs(sub_boxes[2][1] - sub_boxes[1][1]) > 7:
            flag = True
            i = 0
            while flag and i < len(subtitle):
                flag = is_chinese(subtitle[i])
                i+=1
            if flag:
                return True, box2int(sub_boxes), subtitle
    return False, None, None

    
def get_srt(input_video):

    
    srt_file = input_video.split(".m")[0] + ".srt"
    print(input_video,srt_file)
    #ocr = PaddleOCR(use_gpu=True, show_log = False,lang="ch")
    subsampling = 3
    similarity_thresh = 0.8
    cap = cv2.VideoCapture(input_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h = h//4
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('Video info w: {}, h: {}, count: {}, fps: {}'.format(
        w, h, count, fps))

    cur = 0
    detected = False
    box = None
    content = ''
    start = 0
    ref_gray_image = None
    subs = []

    def _add_subs(end):
        #print('New subtitle {} {} {}'.format(start / fps, end / fps, content))
        subs.append(
            srt.Subtitle(
                index=0,
                start=datetime.timedelta(seconds=start / fps),
                end=datetime.timedelta(seconds=end / fps),
                content=content.strip(),
            ))

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if detected:
                _add_subs(cur)
            break
        cur += 1
        if cur % subsampling != 0:
            continue
        frame = frame[-h:, :, :] #h,w,channel
        #print(np.shape(frame))
        if detected:
            # Compute similarity to reference subtitle area, if the result is
            # bigger than thresh, it's the same subtitle, otherwise, there is
            # changes in subtitle area
            boxes_h = sorted([box[1][1],box[2][1]])
            boxes_v = sorted([box[0][0],box[1][0]])
            #hyp_gray_image = frame[box[1][1]:box[2][1], box[0][0]:box[1][0], :]
            hyp_gray_image = frame[boxes_h[0]:boxes_h[1], boxes_v[0]:boxes_v[1], :]
            hyp_gray_image = cv2.cvtColor(hyp_gray_image, cv2.COLOR_BGR2GRAY)
            similarity = structural_similarity(hyp_gray_image, ref_gray_image)
            if similarity > similarity_thresh:  # the same subtitle
                continue
            else:
               # Record current subtitle
                _add_subs(cur - subsampling)
                detected = False
        else:
            # Detect subtitle area
            paddle_results = ppocr_v3.predict(frame)
            paddle_boxes = np.reshape(paddle_results.boxes,(-1,4,2)).tolist()
            paddle_text = paddle_results.text
            paddle_score = paddle_results.rec_scores
            ocr_results = [None]
            if len(paddle_boxes) > 0 and len(paddle_boxes) == len(paddle_text):
                ocr_results = list()
                for i in range(len(paddle_boxes)):
                    if paddle_score[i] > 0.5:
                        ocr_results.append([paddle_boxes[i],paddle_text[i]])
        
            
            if not ocr_results==[None]:
                detected, box, content = detect_subtitle_area(ocr_results, h, w)
                #print(detected,box,content)
                if detected:
                    start = cur
                    boxes_h = sorted([box[1][1],box[2][1]])
                    boxes_v = sorted([box[0][0],box[1][0]])
                    #ref_gray_image = frame[box[1][1]:box[2][1],
                    #                       box[0][0]:box[1][0], :]
                    ref_gray_image =  frame[boxes_h[0]:boxes_h[1], boxes_v[0]:boxes_v[1], :]
                    ref_gray_image = cv2.cvtColor(ref_gray_image,
                                              cv2.COLOR_BGR2GRAY)
            
            
    cap.release()

    # Write srt file
    with open(srt_file, 'w', encoding='utf8') as fout:
        fout.write(srt.compose(subs))

    print("start demucs")
    #os.system(f"CUDA_VISIBLE_DEVICES=1 demucs --two-stems=vocals {input_video} -o ./data/")
    os.system(f"demucs --two-stems=vocals {input_video} -o ./data/")
    
    basename = input_video.split("/")[-1].split(".")[0]
    vocals_path = input_video.split(".m")[0] + "_vocals.wav"
    print("start ffmpeg")
    os.system(f"ffmpeg -loglevel quiet -i ./data/htdemucs/{basename}/vocals.wav -ar 48000 -ac 1 -acodec pcm_s16le {vocals_path}")

def get_no_srt(input_video):
    print("start demucs")
    os.system(f"CUDA_VISIBLE_DEVICES=1 demucs --two-stems=vocals {input_video} -o ./data/")
    
    basename = input_video.split("/")[-1].split(".")[0]
    vocals_path = input_video.split(".m")[0] + "_vocals.wav"
    print("start ffmpeg")
    os.system(f"ffmpeg -loglevel quiet -i ./data/htdemucs/{basename}/vocals.wav -ar 48000 -ac 1 -acodec pcm_s16le {vocals_path}")

if __name__ == "__main__":
    option = fd.RuntimeOption()
    option.use_gpu()
    option.use_paddle_infer_backend()
    
    
    mp4_files = glob.glob("./data/raw_data/*.mp4")
    mp4_files.extend(glob.glob("./data/raw_data/*.mkv"))
    mp4_files = sorted(mp4_files)
    #mp4_files = sorted(glob.glob("./data/raw_data/*.mkv"))
    #mp4_files.extend(sorted(glob.glob("./data/raw_data/4*.mp4")))
    #mp4_files.extend(sorted(glob.glob("./data/raw_data/5*.mp4")))
    #mp4_files = ["./data/raw_data/19.mp4"]
    thread_num = 4
    #thread_num = min(len(mp4_files),os.cpu_count()//2)

    for i in range(0,len(mp4_files),thread_num):
        process_files = mp4_files[i:min(i+thread_num, len(mp4_files))]
        
        real_thread_num = len(process_files)
        print(real_thread_num)
        with Pool(
                real_thread_num,
                initializer=load_model,
                initargs=(option,)) as pool:
            pool.map(get_srt, process_files)
            #pool.map(get_no_srt, process_files)
