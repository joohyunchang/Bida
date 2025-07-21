import argparse
import subprocess
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path
from tqdm import tqdm
from functools import partial

def ffmpeg_extraction(input_video, output_sound, start_timestamp, stop_timestamp, sample_rate):
    ffmpeg_command = ['ffmpeg', '-loglevel', 'error', '-y', '-i', input_video, '-ss', start_timestamp, '-to', stop_timestamp, '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', sample_rate, output_sound]
    subprocess.call(ffmpeg_command)
    

def trim_video(input_video, output_video, start_timestamp, stop_timestamp):
    scale_expression = "scale='if(gte(iw,ih),256*iw/min(iw,ih),-1)':'if(lt(iw,ih),256*ih/min(iw,ih),-1)'"
    # ffmpeg_command = ['ffmpeg', '-loglevel', 'error', '-y', '-i', input_video, '-vf', scale_expression, '-ss', start_timestamp, '-to', stop_timestamp, '-c:v', 'h264', '-c:a', 'copy', output_video]
    # ffmpeg_command = ['ffmpeg', '-loglevel', 'error', '-y', '-i', input_video, '-vf', scale_expression, '-ss', start_timestamp, '-to', stop_timestamp, '-c:a', 'copy', output_video]
    ffmpeg_command = ['ffmpeg', '-loglevel', 'error', '-y', '-i', input_video, '-ss', start_timestamp, '-to', stop_timestamp, '-c', 'copy', output_video]
    subprocess.call(ffmpeg_command)
    
def trim_video_by_frame(input_video, output_video, start_frame, stop_frame):
    ffmpeg_command = ['ffmpeg', '-y' ,'-i', input_video,
                      '-vf', f'select=between(n\\,{start_frame}\\,{stop_frame})',
                      '-vsync', 'vfr', output_video]

    subprocess.call(ffmpeg_command)
    
def process_video(anno_id, vid_id, video_id, start, stop, videos_dir, output_video_dir, audios_dir, output_audio_dir):
    try:
        video_path = os.path.join(videos_dir, vid_id, video_id + '.mp4')
        audio_path = os.path.join(audios_dir, vid_id, video_id + '.mp4')
        output_video_path = os.path.join(output_video_dir, anno_id + '.mp4')
        output_audio_path = os.path.join(output_audio_dir, anno_id + '.wav')

        # trim_video(video_path, output_video_path, start, stop)
        ffmpeg_extraction(audio_path, output_audio_path, start, stop, '24000')
    except Exception as e:
        print(f"Error processing file: {video_path}")
        print(e)


if __name__ == '__main__' and False:
    parser = argparse.ArgumentParser()
    parser.add_argument('step', type=int, default=0, help='Rate to resample audio')
    # parser.add_argument('videos_dir', help='Directory of EPIC videos with audio')
    # parser.add_argument('output_dir', help='Directory of EPIC videos with audio')
    # parser.add_argument('type', default='mp4', help='Video type')
    # parser.add_argument('--sample_rate', default='24000', help='Rate to resample audio')
    args = parser.parse_args()
    
    if False:
        anno_file = '/data/joohyun7u/project/CAST/dataset/epic_sounds/EPIC_Sounds_train.csv'
        # anno_file = '/data/joohyun7u/project/CAST/dataset/epic_sounds/EPIC_Sounds_validation.csv'
        annotation = pd.read_csv(anno_file, header=0, delimiter=',')
        videos_dir = '/data/datasets/epic_resized/EPIC-KITCHENS/'
        audios_dir = '/data/datasets/epickitchens/EPIC-KITCHENS/'
        output_dir = '/data/datasets/epic_sound'
    else:
        anno_file = '/data/joohyun7u/project/CAST/dataset/hd_epic/HD_EPIC_Narrations.json'
        annotation = pd.read_json(anno_file, lines=True)
        videos_dir = '/data/dataset/HD-EPIC/Videos'
        audios_dir = '/data/dataset/HD-EPIC/Videos'
        output_dir = '/data/dataset/HD-EPIC/hd-epic-trimmed'
    
    stride = 5000
    a = int(args.step)
    start_idx = a * stride
    end_idx = (a+1) * stride
    real_total = len(annotation)
    start_idx = len(annotation) if start_idx > len(annotation) else start_idx
    end_idx = len(annotation) if end_idx > len(annotation) else end_idx
    print('sep', start_idx, ' to ', end_idx)
    annotation = annotation.iloc[start_idx:end_idx]
    
    # existing_wavs = set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(output_dir, 'wav')))
    # annotation = annotation[~annotation['unique_narration_id'].astype(str).isin(existing_wavs)].reset_index(drop=True)
    # start_idx, end_idx = 0, 200
    # print('left ', len(annotation), ' from ', real_total)
    # end_idx = len(annotation) if end_idx > len(annotation) else end_idx
    # annotation = annotation.iloc[start_idx:end_idx]
    output_video_dir = os.path.join(output_dir,'video')
    output_audio_dir = os.path.join(output_dir,'wav')

    # if not os.path.exists(output_video_dir):
    #     os.makedirs(output_video_dir)
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)

    # 병렬 처리를 위한 ThreadPoolExecutor 설정
    # with ProcessPoolExecutor(max_workers=64) as executor:
    #     for anno_id, vid_id, video_id, start, stop in zip(annotation['annotation_id'], annotation['participant_id'], annotation['video_id'], annotation['start_timestamp'], annotation['stop_timestamp']):
    #         executor.submit(process_video, anno_id, vid_id, video_id, start, stop, videos_dir, output_video_dir, output_audio_dir)
    
    tasks = []
    total = len(annotation)
    start_time = time.time()
    temp_time = time.time()
    # with ProcessPoolExecutor(max_workers=250) as executor:
    #     # 작업 제출
    #     for idx, row in annotation.iterrows():
    #         task = executor.submit(process_video, row['annotation_id'], row['participant_id'], row['video_id'], row['start_timestamp'], row['stop_timestamp'], videos_dir, output_video_dir, audios_dir, output_audio_dir)
    #         tasks.append(task)

    #     # 진행 상태 추적
    #     for i, task in enumerate(as_completed(tasks), 1):
    #         if i % 100 == 0:
    #             print(f"진행 상태: {i}/{total} ({i/total*100:.2f}%)  {time.time()-temp_time:.3f}")
    #             temp_time = time.time()

    # 작업 제출
    for idx, row in annotation.iterrows():
        # process_video(row['annotation_id'], row['participant_id'], row['video_id'], row['start_timestamp'], row['stop_timestamp'], videos_dir, output_video_dir, audios_dir, output_audio_dir)
        # print(f"진행 상태: {idx}/{total} ({idx/total*100:.2f}%)  {time.time()-temp_time:.3f},   {row['annotation_id']}")
        process_video(row['unique_narration_id'], row['participant_id'], row['video_id'], str(row['start_timestamp']), str(row['end_timestamp']), videos_dir, output_video_dir, audios_dir, output_audio_dir)
        print(f"진행 상태: {idx}/{total}/{real_total} ({(idx-start_idx)/total*100:.2f}%)  {time.time()-temp_time:.3f},   {row['unique_narration_id']}")
        temp_time = time.time()
        
    print('total time : ', time.time()-start_time)

def process_single_file_with_video(row_tuple, source_root, destination_root):
    """하나의 비디오 클립을 처리하는 작업자 함수"""
    idx, row = row_tuple
    
    video_path = source_root / row['participant_id'] / (row['video_id'] + '.mp4')
    
    if not video_path.exists():
        return f"Skipped: {video_path} not found."

    # 최종 결과물은 비디오 파일(.mp4)
    output_path = destination_root / (row['video_id'] + f'-{row['start_sample']}' + '.mp4')
    
    try:
        command = [
            'ffmpeg',
            '-ss', str(row['start_timestamp']),
            '-i', str(video_path),
            '-to', str(row['stop_timestamp']),
            '-vf', "scale='if(gte(iw,ih),-1,256)':'if(gte(iw,ih),256,-1)'",
            '-c:v', 'libopenh264',
            # '-preset', 'ultrafast', # 속도 우선
            '-acodec', 'aac',       # mp4와 호환되는 오디오 코덱
            # '-ac', '1',
            '-ar', '24000',
            '-y',
            '-loglevel', 'error',
            str(output_path)
        ]
        subprocess.run(command, check=True, capture_output=True)
        return f"Processed: {output_path.name}"
    except subprocess.CalledProcessError as e:
        return f"Failed: {output_path.name} | Error: {e.stderr.decode()}"
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', type=int, default=0, help='Chunk step for processing')
    args = parser.parse_args()
    
    # 경로 변수를 메인 로직에서 관리
    anno_file = Path('/data/joohyun7u/project/CAST/dataset/hd_epic_audio_sounds/HD_EPIC_Sounds.csv')
    SOURCE_PATH = Path("/data/dataset/HD-EPIC/Videos")
    DESTINATION_PATH = Path("/data/dataset/HD-EPIC/hd-epic-sounds-trimmed") # 목적에 맞게 경로 이름 변경
    
    os.makedirs(DESTINATION_PATH, exist_ok=True)
    
    print("Loading annotation file...")
    anno_df = pd.read_csv(anno_file)
    
    # # 기존에 처리된 mp4 파일명 집합 생성 (video_id+start_sample 조합)
    # existing_mp4 = set(os.path.splitext(f)[0] for f in os.listdir(DESTINATION_PATH))
    # anno_df['mp4_name'] = anno_df['video_id'].astype(str) + '-' + anno_df['start_sample'].astype(str)
    # anno_df = anno_df[~anno_df['mp4_name'].isin(existing_mp4)].reset_index(drop=True)
    
    stride = 2600
    start_idx = args.step * stride
    end_idx = min((args.step + 1) * stride, len(anno_df))
    
    if start_idx >= len(anno_df):
        print("Start index is out of bounds. No files to process.")
        exit()
        
    print(f'Processing chunk {args.step}: from index {start_idx} to {end_idx}')
    df_chunk = anno_df.iloc[start_idx:end_idx]

    # num_workers = os.cpu_count() 
    num_workers = 8
    print(f"Starting parallel processing with {num_workers} workers...")
    
    # 🚀 functools.partial을 사용하여 함수의 일부 인자를 고정
    # 이제 process_func는 row_tuple 하나만 인자로 받는 함수처럼 동작합니다.
    process_func = partial(
        process_single_file_with_video, 
        source_root=SOURCE_PATH, 
        destination_root=DESTINATION_PATH
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = list(df_chunk.iterrows())
        results = list(tqdm(executor.map(process_func, tasks), total=len(tasks)))
    
    print("\nProcessing finished.")
    
    print("--- Checking for failed or skipped files ---")
    failed_count = 0
    for res in results:
        # 반환된 메시지에 'Failed' 또는 'Skipped'가 포함되어 있으면 출력
        if "Failed" in res or "Skipped" in res:
            print(res)
            failed_count += 1
            
    if failed_count == 0:
        print("All files processed successfully.")
    else:
        print(f"\nTotal {failed_count} files failed or were skipped.")