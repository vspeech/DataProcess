import json
import os
import re
import subprocess

# --- 配置 ---
# !!! 重要: 請在這裡修改你要處理的 JSON 檔案名稱 !!!
JSON_FILE = 'taigi_days.json' 
DOWNLOAD_DIR = 'taiwan' # 所有音檔都會被下載到這個資料夾
BASE_URL = 'https://channelplus.ner.gov.tw/api/audio/'

# --- 主邏輯 ---

# 1. 檢查 JSON 檔案是否存在
if not os.path.exists(JSON_FILE):
    print(f"錯誤: 找不到 {JSON_FILE} 檔案。請確保檔名正確，且它和腳本在同一個目錄下。")
    exit()

# 2. 如果下載目錄不存在，則創建
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
    print(f"已創建下載目錄: {DOWNLOAD_DIR}")

# 3. 讀取並解析 JSON 檔案
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 4. 智慧判斷節目列表的鍵名 (是 'rows' 還是 'episodes')
episode_list = None
if 'rows' in data:
    episode_list = data['rows']
elif 'episodes' in data:
    episode_list = data['episodes']

if episode_list is None:
    print(f"錯誤: 在 {JSON_FILE} 中找不到 'rows' 或 'episodes' 列表。請檢查檔案內容。")
    exit()

# 5. 遍歷每一集播客並開始下載
total_episodes = len(episode_list)
print(f"在 '{JSON_FILE}' 中共找到 {total_episodes} 集節目。")
print("-" * 30)

for i, episode in enumerate(episode_list):
    print(f"正在處理第 {i+1} / {total_episodes} 集...")

    try:
        # 提取需要的信息
        # 優先使用 audio.name 作為檔名，如果沒有，再用外層的 name
        episode_title = episode['audio'].get('name') or episode.get('name', '未知標題')
        audio_key = episode['audio']['key']

        # 清理檔名中的非法字元，並去除.mp3副檔名 (wget會自動處理)
        clean_title = re.sub(r'[\\/*?:"<>|]', "", episode_title)
        if clean_title.endswith('.mp3'):
            clean_title = clean_title[:-4]
        
        # 拼接成最終的檔案路徑和下載連結
        output_path = os.path.join(DOWNLOAD_DIR, f"{clean_title}.mp3")
        download_url = f"{BASE_URL}{audio_key}"

        # 檢查檔案是否已存在，如果存在則跳過
        if os.path.exists(output_path):
            print(f"檔案 '{clean_title}.mp3' 已存在，跳過下載。")
            print("-" * 30)
            continue
        
        print(f"開始下載: {clean_title}")
        
        # 建立並執行 wget 命令
        command = ['wget', '-O', output_path, download_url]
        subprocess.run(command, check=True, capture_output=True, text=True) # capture_output讓過程更乾淨
        
        print(f"下載成功: {output_path}")

    except (KeyError, TypeError) as e:
        print(f"警告: 這一集的資訊不完整 (缺少 audio 或 key)，已跳過。錯誤: {e}")
    except subprocess.CalledProcessError as e:
        print(f"錯誤: 下載 '{clean_title}' 失敗。返回碼: {e.returncode}")
        print(f"wget 錯誤訊息: {e.stderr}")
    except FileNotFoundError:
        print("致命錯誤: 找不到 'wget' 命令。請確保 wget 已安裝並已添加到系統路徑中。")
        exit()
    except Exception as e:
        print(f"發生未知錯誤: {e}")
    
    print("-" * 30)

print("\n所有下載任務已處理完畢！")