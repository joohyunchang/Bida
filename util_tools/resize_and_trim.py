# import subprocess
# import pandas as pd
# from pathlib import Path
# import os
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# from tqdm import tqdm

# def resize_video(row_tuple, source_root, destination_root):
#     _, row = row_tuple
#     input_path = source_root / row['participant_id'] / (row['video_id'] + '.mp4')
#     output_path = destination_root / (row['video_id'] + '_resized.mp4')

#     if not input_path.exists():
#         return f"Skipped: {input_path} not found."
#     if output_path.exists() and output_path.stat().st_size > 1024:
#         return f"Skipped (already resized): {output_path.name}"

#     try:
#         command = [
#             'ffmpeg',
#             '-i', str(input_path),
#             '-vf', "scale='if(gte(iw,ih),-1,256)':'if(gte(iw,ih),256,-1)'",
#             '-c:v', 'libopenh264',
#             '-acodec', 'aac',
#             '-ar', '24000',
#             '-y',
#             '-loglevel', 'error',
#             str(output_path)
#         ]
#         subprocess.run(command, check=True, capture_output=True)
#         return f"Resized: {output_path.name}"
#     except subprocess.CalledProcessError as e:
#         return f"Failed: {output_path.name} | Error: {e.stderr.decode()}"

# if __name__ == '__main__':
#     anno_file = Path('/data/joohyun7u/project/CAST/dataset/hd_epic_audio_sounds/HD_EPIC_Sounds.csv')
#     SOURCE_PATH = Path("/data/dataset/HD-EPIC/Videos")
#     DEST_PATH = Path("/data/dataset/HD-EPIC/hd-epic-resized")
#     os.makedirs(DEST_PATH, exist_ok=True)
#     anno_df = pd.read_csv(anno_file)
#     unique_videos = anno_df[['participant_id', 'video_id']].drop_duplicates().reset_index(drop=True)

#     num_workers = 88  # 환경에 맞게 조정
#     process_func = partial(resize_video, source_root=SOURCE_PATH, destination_root=DEST_PATH)
#     tasks = list(unique_videos.iterrows())
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         results = list(tqdm(executor.map(process_func, tasks), total=len(tasks)))
#     for msg in results:
#         print(msg)







# import subprocess
# import pandas as pd
# from pathlib import Path
# import os
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# from tqdm import tqdm

# def trim_clip(row_tuple, resized_root, destination_root):
#     _, row = row_tuple
#     input_path = resized_root / (row['video_id'] + '_resized.mp4')
#     output_path = destination_root / (row['video_id'] + f'-{row["start_sample"]}.mp4')

#     if not input_path.exists():
#         return f"Skipped: {input_path} not found."
#     if output_path.exists() and output_path.stat().st_size > 1024:
#         return f"Skipped (already trimmed): {output_path.name}"

#     try:
#         command = [
#             'ffmpeg',
#             '-ss', str(row['start_timestamp']),
#             '-i', str(input_path),
#             '-to', str(row['stop_timestamp']),
#             '-c:v', 'libopenh264',    # 또는 libx264 등
#             '-c:a', 'aac',
#             '-ar', '24000',
#             '-y',
#             '-loglevel', 'error',
#             str(output_path)
#         ]
#         subprocess.run(command, check=True, capture_output=True)
#         return f"Trimmed: {output_path.name}"
#     except subprocess.CalledProcessError as e:
#         return f"Failed: {output_path.name} | Error: {e.stderr.decode()}"

# if __name__ == '__main__':
#     anno_file = Path('/data/joohyun7u/project/CAST/dataset/hd_epic_audio_sounds/HD_EPIC_Sounds.csv')
#     RESIZED_PATH = Path("/data/dataset/HD-EPIC/hd-epic-resized")
#     DEST_PATH = Path("/data/dataset/HD-EPIC/hd-epic-sounds-trimmed3")
#     os.makedirs(DEST_PATH, exist_ok=True)
#     anno_df = pd.read_csv(anno_file)

#     num_workers = 96  # 환경에 맞게 조정
#     process_func = partial(trim_clip, resized_root=RESIZED_PATH, destination_root=DEST_PATH)
#     tasks = list(anno_df.iterrows())
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         results = list(tqdm(executor.map(process_func, tasks), total=len(tasks)))
#     for msg in results:
#         print(msg)


import subprocess
import pandas as pd
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from datetime import datetime

# Duration 계산 (초)
def ts2float(t):
    if '.' not in t: t += '.0'
    dt = datetime.strptime(t, '%H:%M:%S.%f')
    return dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1e6
    
def trim_clip(row_tuple, resized_root, destination_root, timeout_sec=180, min_duration=1/30):
    _, row = row_tuple
    input_path = resized_root / (row['video_id'] + '_resized.mp4')
    output_path = destination_root / (row['video_id'] + f'-{row["start_sample"]}-{row["stop_sample"]}.mp4')

    if not input_path.exists():
        return False, f"Skipped: {input_path} not found."
    if output_path.exists() and output_path.stat().st_size > 1024:
        return True, f"Skipped (already trimmed): {output_path.name}"

    duration = ts2float(row['stop_timestamp']) - ts2float(row['start_timestamp'])
    # 최소 duration 보정 (예: 1/30초)
    if duration < min_duration:
        duration = min_duration
 
    try:
        command = [
            'ffmpeg',
            '-ss', str(row['start_timestamp']),
            '-i', str(input_path),
            # '-to', str(row['stop_timestamp']),
            '-t', f"{duration:.3f}",
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-ar', '24000',
            '-y',
            '-loglevel', 'error',
            str(output_path)
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=timeout_sec)
        return True, f"Trimmed: {output_path.name}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout: {output_path.name} exceeded {timeout_sec}s"
    except subprocess.CalledProcessError as e:
        return False, f"Failed: {output_path.name} | Error: {e.stderr.decode()}"

if __name__ == '__main__':
    anno_file = Path('/data/joohyun7u/project/CAST/dataset/hd_epic_sounds/HD_EPIC_Sounds.csv')
    RESIZED_PATH = Path("/data/dataset/HD-EPIC/hd-epic-resized")
    DEST_PATH = Path("/data/dataset/HD-EPIC/hd-epic-sounds-trimmed3")
    os.makedirs(DEST_PATH, exist_ok=True)
    anno_df = pd.read_csv(anno_file)

    num_workers = 96  # 환경에 맞게 조정
    timeout_sec = 3600  # 구간 길이에 따라 늘릴 수 있음
    process_func = partial(trim_clip, resized_root=RESIZED_PATH, destination_root=DEST_PATH, timeout_sec=timeout_sec)
    tasks = list(anno_df.iterrows())

    with open("trim_success.log", "a") as f_success, open("trim_fail.log", "a") as f_fail:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_func, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures)):
                try:
                    success, msg = f.result()
                    print(msg)
                    if success:
                        f_success.write(msg + "\n")
                        f_success.flush()
                    else:
                        f_fail.write(msg + "\n")
                        f_fail.flush()
                except Exception as e:
                    err = f"Unknown error: {e}"
                    print(err)
                    f_fail.write(err + "\n")
                    f_fail.flush()
