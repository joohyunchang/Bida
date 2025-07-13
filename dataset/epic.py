import os
from statistics import NormalDist
import numpy as np
import torch
from torchvision import transforms
from util_tools.random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import util_tools.video_transforms as video_transforms 
import util_tools.volume_transforms as volume_transforms
from util_tools.audio_transforms import Spectrogram
from util_tools.audio_transforms import save_spectrogram_npy
import torchaudio
import random
import torch

class EpicVideoClsDataset(Dataset):
     def __init__(self, anno_path, data_path, mode='train', clip_len=8,
               crop_size=224, short_side_size=256, new_height=256,
               new_width=340, keep_aspect_ratio=True, num_segment=1,
               num_crop=1, test_num_segment=10, test_num_crop=3, args=None, audio_path=None):
          self.anno_path = anno_path
          self.data_path = data_path
          self.audio_path = args.audio_path
          self.mode = mode
          self.clip_len = clip_len
          self.crop_size = crop_size
          self.short_side_size = short_side_size
          self.new_height = new_height
          self.new_width = new_width
          self.keep_aspect_ratio = keep_aspect_ratio
          self.num_segment = num_segment
          self.test_num_segment = test_num_segment
          self.num_crop = num_crop
          self.test_num_crop = test_num_crop
          self.args = args
          self.aug = False
          self.rand_erase = False
          self.disable_video = args.disable_video
          if self.mode in ['train']:
               self.aug = True
               if self.args.reprob > 0:
                    self.rand_erase = True
          if VideoReader is None:
               raise ImportError("Unable to import `decord` which is required to read videos.")
          if self.audio_path is not None:
               self.audio_extension = '.wav'
               self.audio_type = args.audio_type
               self.realtime_audio = args.realtime_audio
               self.autosave_spec = args.autosave_spec
               self.data_set = 'EPIC_split'
               if (mode == 'train') and getattr(args, 'add_noise', None): 
                    self.spectrogram = Spectrogram(num_segment, args.audio_height, args.audio_width, n_fft=2048, process_type=args.process_type, noisereduce=args.noisereduce, specnorm=args.specnorm, noise=True)
               else:
                    self.spectrogram = Spectrogram(num_segment, args.audio_height, args.audio_width, n_fft=2048, process_type=args.process_type, noisereduce=args.noisereduce, specnorm=args.specnorm)
          
          import pandas as pd
          import pickle
          cleaned = pd.read_csv(self.anno_path, header=0, delimiter=',')
          # if self.mode == 'train':
          #      self.dataset_samples = list(cleaned.values[:, 0])[:101]
          # else:
          self.dataset_samples = list(cleaned.values[:, 0])
          verb_label_array = list(cleaned.values[:, 1]) # verb
          noun_label_array = list(cleaned.values[:, 2]) # noun
          action_label_array = list(cleaned.values[:, 3]) # action
          self.audio_samples = {cleaned.iloc[i, 0]: cleaned.iloc[i, 12:14] for i in range(len(cleaned))}
          self.narration_array = {cleaned.iloc[i, 0]: eval(cleaned.iloc[i, 9]) for i in range(len(cleaned))} if args.narration else None
          self.narration_array = {cleaned.iloc[i, 0]: eval(cleaned.iloc[i, 14]) for i in range(len(cleaned))} if args.class_narration else self.narration_array
          self.label_array = np.stack((noun_label_array, verb_label_array, action_label_array), axis=1) # label [noun, verb] sequence
          
          if  (mode == 'train'):
               pass
          
          elif (mode == 'validation'):
               self.data_transform = video_transforms.Compose([
                    video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                    video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
               ])
          elif (mode == 'test'):
               self.data_resize = video_transforms.Compose([
                    video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
               ])
               self.data_transform = video_transforms.Compose([
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
               ])
               self.test_seg = []
               self.test_dataset = []
               self.test_label_array = []
               for ck in range(self.test_num_segment):
                    for cp in range(self.test_num_crop):
                         for idx in range(len(self.label_array)):
                              sample_label = self.label_array[idx]
                              self.test_label_array.append(sample_label)
                              self.test_dataset.append(self.dataset_samples[idx])
                              self.test_seg.append((ck, cp))
                              
     def __getitem__(self, index):
          idx, all_idx = np.zeros(0), np.zeros(0)
          if self.mode == 'train':
               args = self.args
               scale_t = 1
               # caption = random.choice(self.narration_array[self.dataset_samples[index]]).strip('#C').strip('#c').strip('#0') if self.narration_array is not None else None
               caption = random.choice(self.narration_array[self.dataset_samples[index]]) if self.narration_array is not None else None
               
               if self.audio_path is not None:
                    audio_trim_path = os.path.join(self.audio_path,'../spec', self.audio_type, self.dataset_samples[index] + '.npy')
                    audio_trim_path = audio_trim_path.replace("single", "stacks") if self.audio_type == 'single' else audio_trim_path
                    audio_trim_path = audio_trim_path.replace("singles", "stackss") if self.audio_type == 'singles' else audio_trim_path
                    if os.path.exists(audio_trim_path) and not self.realtime_audio:
                         spec = self.spectrogram.loadaudiofromfile(audio_trim_path, self.audio_type)
                         if args.spec_augment:
                              spec = self.spectrogram.spec_augment(spec)
                    else:
                         # audio_id = '_'.join(self.dataset_samples[index].split('_')[:-1])
                         # audio_sample = os.path.join(self.audio_path, 'wav', audio_id + self.audio_extension)
                         audio_sample = os.path.join(self.audio_path, self.dataset_samples[index] + self.audio_extension)
                         start_frame = self.audio_samples[self.dataset_samples[index]]['start_frame']
                         end_frame = self.audio_samples[self.dataset_samples[index]]['stop_frame']
                         try:
                              spec, idx = self.spectrogram.loadaudio(audio_sample, start_frame, end_frame, audio_centra=random.random(), audio_type=self.audio_type, data_set=self.data_set, mode=self.mode, return_index=True)
                              if not self.realtime_audio and self.autosave_spec:
                                   if not os.path.exists(os.path.join(self.audio_path,'spec', self.audio_type)):
                                        os.makedirs(os.path.join(self.audio_path,'spec', self.audio_type), exist_ok=True)
                                   try:
                                        save_spec = spec[0].detach()
                                        save_spectrogram_npy(audio_trim_path, save_spec)
                                   except:
                                        pass
                              if args.spec_augment:
                                   spec = self.spectrogram.spec_augment(spec)
                         except Exception as e:
                              warnings.warn("audio {} not correctly loaded during training, {}".format(audio_sample, self.dataset_samples[index]))
                              warnings.warn(e)
               else:
                    spec = {}
               
               sample = self.dataset_samples[index] + '.mp4'
               sample = os.path.join(self.data_path, sample)
               if self.disable_video:
                    return torch.tensor([1]), self.label_array[index], sample.split("/")[-1].split(".")[0], spec, caption, all_idx
               buffer, all_idx = self.loadvideo_decord(sample, sample_rate_scale=scale_t, return_index=True) # T H W C
               
               if len(buffer) == 0:
                    while len(buffer) == 0:
                         warnings.warn("video {} not correctly loaded during training".format(sample))
                         index = np.random.randint(self.__len__())
                         sample = self.dataset_samples[index] + '.mp4'
                         sample = os.path.join(self.data_path, sample)
                         buffer, all_idx = self.loadvideo_decord(sample, sample_rate_scale=scale_t, return_index=True)
                         
               if args.num_sample > 1:
                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(args.num_sample):
                         new_frames = self._aug_frame(buffer, args)
                         label = self.label_array[index]
                         frame_list.append(new_frames)
                         label_list.append(label)
                         index_list.append(index)
                    return frame_list, label_list, index_list, {}
               else:
                    buffer = self._aug_frame(buffer, args)
               all_idx = np.concatenate((idx, all_idx))
               return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], spec, caption, all_idx
          
          elif self.mode == 'validation':
               # caption = random.choice(self.narration_array[self.dataset_samples[index]]).strip('#C').strip('#c').strip('#0') if self.narration_array is not None else None
               caption = random.choice(self.narration_array[self.dataset_samples[index]]) if self.narration_array is not None else None
               
               if self.audio_path is not None:
                    audio_trim_path = os.path.join(self.audio_path,'../spec', self.audio_type, self.dataset_samples[index] + '.npy')
                    audio_trim_path = audio_trim_path.replace("single", "stacks") if self.audio_type == 'single' else audio_trim_path
                    audio_trim_path = audio_trim_path.replace("singles", "stackss") if self.audio_type == 'singles' else audio_trim_path
                    if os.path.exists(audio_trim_path) and not self.realtime_audio:
                         spec = self.spectrogram.loadaudiofromfile(audio_trim_path, self.audio_type)
                    else:
                         # audio_id = '_'.join(self.dataset_samples[index].split('_')[:-1])
                         # audio_sample = os.path.join(self.audio_path, 'wav', audio_id + self.audio_extension)
                         audio_sample = os.path.join(self.audio_path, self.dataset_samples[index] + self.audio_extension)
                         start_frame = self.audio_samples[self.dataset_samples[index]]['start_frame']
                         end_frame = self.audio_samples[self.dataset_samples[index]]['stop_frame']
                         spec, idx = self.spectrogram.loadaudio(audio_sample, start_frame, end_frame, audio_type=self.audio_type, data_set=self.data_set, return_index=True)
                         if not self.realtime_audio and self.autosave_spec:
                              try:
                                   save_spec = spec[0]
                                   save_spectrogram_npy(audio_trim_path, save_spec)
                              except:
                                   pass
               else:
                    spec = {}
               
               sample = self.dataset_samples[index] + '.mp4'
               sample = os.path.join(self.data_path, sample)
               if self.disable_video:
                    return torch.tensor([1]), self.label_array[index], sample.split("/")[-1].split(".")[0], spec, caption, all_idx
               buffer, all_idx = self.loadvideo_decord(sample, return_index=True)
               
               if len(buffer) == 0:
                    while len(buffer) == 0:
                         warnings.warn("video {} not correctly loaded during validation".format(sample))
                         index = np.random.randint(self.__len__())
                         sample = self.dataset_samples[index] + '.mp4'
                         sample = os.path.join(self.data_path, sample)
                         buffer, all_idx = self.loadvideo_decord(sample, return_index=True)
               buffer = self.data_transform(buffer)
               all_idx = np.concatenate((idx, all_idx))
               return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], spec, caption, all_idx
          
          elif self.mode == 'test':
               # caption = random.choice(self.narration_array[self.dataset_samples[index]]).strip('#C').strip('#c').strip('#0') if self.narration_array is not None else None
               caption = random.choice(self.narration_array[self.dataset_samples[index]]) if self.narration_array is not None else None
               chunk_nb, split_nb = self.test_seg[index]
               if self.audio_path is not None:
                    audio_trim_path = os.path.join(self.audio_path,'../spec', self.audio_type, self.test_dataset[index] + '.npy')
                    audio_trim_path = audio_trim_path.replace("single", "stacks") if self.audio_type == 'single' else audio_trim_path
                    audio_trim_path = audio_trim_path.replace("singles", "stackss") if self.audio_type == 'singles' else audio_trim_path
                    if os.path.exists(audio_trim_path) and not self.realtime_audio:
                         spec = self.spectrogram.loadaudiofromfile(audio_trim_path, self.audio_type)
                    else:
                         # audio_id = '_'.join(self.test_dataset[index].split('_')[:-1])
                         # audio_sample = os.path.join(self.audio_path, 'wav', audio_id + self.audio_extension)
                         audio_sample = os.path.join(self.audio_path, self.test_dataset[index] + self.audio_extension)
                         start_frame = self.audio_samples[self.test_dataset[index]]['start_frame']
                         end_frame = self.audio_samples[self.test_dataset[index]]['stop_frame']
                         # (2*chunk_nb+1)/(self.test_num_segment * 2)
                         try:
                              spec, idx = self.spectrogram.loadaudio(audio_sample, start_frame, end_frame, audio_centra=(self.test_num_crop*chunk_nb+split_nb+3)/(self.test_num_segment * self.test_num_crop+6), audio_type=self.audio_type, data_set=self.data_set, return_index=True)
                         except Exception as e:
                              warnings.warn("audio {} not correctly loaded during testing, {}".format(audio_sample, self.test_dataset[index]))
                              warnings.warn(e)
                              spec = {}
               else:
                    spec = {}
               
               sample = self.test_dataset[index] + '.mp4'
               sample = os.path.join(self.data_path, sample)
               chunk_nb, split_nb = self.test_seg[index]
               buffer, all_idx = self.loadvideo_decord(sample, return_index=True)
               if self.disable_video:
                    return torch.tensor([1]), self.test_label_array[index], sample.split("/")[-1].split(".")[0], chunk_nb, split_nb, spec, caption, all_idx

               while len(buffer) == 0:
                    warnings.warn("video {}, temporal {}, spatial {} not found during testing {}".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb, sample))
                    index = np.random.randint(self.__len__())
                    sample = self.test_dataset[index] + '.mp4'
                    sample = os.path.join(self.data_path, sample)
                    chunk_nb, split_nb = self.test_seg[index]
                    buffer, all_idx = self.loadvideo_decord(sample, return_index=True)

               buffer = self.data_resize(buffer)
               if isinstance(buffer, list):
                    buffer = np.stack(buffer, 0)

               # fix bug (test_crop수가 1 일때 zero division이 발생하는 error debug)
               if self.test_num_crop == 1:
                    spatial_step = 1.0 * (max( buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                        / (self.test_num_crop)
               else:
                    spatial_step = 1.0 * (max( buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                        / (self.test_num_crop - 1)
               sampling_type='sparse'
               if sampling_type == 'sparse':
                    temporal_start = chunk_nb # 0/1
                    all_idx = all_idx[temporal_start::self.test_num_segment]
                    spatial_start = int(split_nb * spatial_step)
                    if buffer.shape[1] >= buffer.shape[2]:
                         buffer = buffer[temporal_start::self.test_num_segment, \
                              spatial_start:spatial_start + self.short_side_size, :, :]
                    else:
                         buffer = buffer[temporal_start::self.test_num_segment, \
                              :, spatial_start:spatial_start + self.short_side_size, :]
               elif sampling_type == 'dense':
                    temporal_step = max(1.0 * (buffer.shape[0] - self.num_segment) \
                                / (self.test_num_segment - 1), 0)
                    temporal_start = int(chunk_nb * temporal_step)
                    all_idx = all_idx[temporal_start:temporal_start + self.num_segment]
                    spatial_start = int(split_nb * spatial_step)
                    if buffer.shape[1] >= buffer.shape[2]:
                         buffer = buffer[temporal_start:temporal_start + self.num_segment, \
                              spatial_start:spatial_start + self.short_side_size, :, :]
                    else:
                         buffer = buffer[temporal_start:temporal_start + self.num_segment, \
                              :, spatial_start:spatial_start + self.short_side_size, :]

               buffer = self.data_transform(buffer)
               all_idx = np.concatenate((idx, all_idx))
               return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                    chunk_nb, split_nb, spec, caption, all_idx
          else:
               raise NameError('mode {} unkown'.format(self.mode))
               
               

     def _aug_frame(self,buffer,args):

          aug_transform = video_transforms.create_random_augment(
               input_size=(self.crop_size, self.crop_size),
               auto_augment=args.aa,
               interpolation=args.train_interpolation,
          )

          buffer = [
               transforms.ToPILImage()(frame) for frame in buffer
          ]

          buffer = aug_transform(buffer)

          buffer = [transforms.ToTensor()(img) for img in buffer]
          buffer = torch.stack(buffer) # T C H W
          buffer = buffer.permute(0, 2, 3, 1) # T H W C 
          
          # T H W C 
          buffer = tensor_normalize(
               buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
          )
          # T H W C -> C T H W.
          buffer = buffer.permute(3, 0, 1, 2)
          # Perform data augmentation.
          scl, asp = (
               [0.08, 1.0],
               [0.75, 1.3333],
          )

          buffer = spatial_sampling(
               buffer,
               spatial_idx=-1,
               min_scale=256,
               max_scale=320,
               crop_size=self.crop_size,
               random_horizontal_flip=False if args.data_set == 'SSV2' else True,
               inverse_uniform_sampling=False,
               aspect_ratio=asp,
               scale=scl,
               motion_shift=False
          )

          if self.rand_erase:
               erase_transform = RandomErasing(
                    args.reprob,
                    mode=args.remode,
                    max_count=args.recount,
                    num_splits=args.recount,
                    device="cpu",
               )
               buffer = buffer.permute(1, 0, 2, 3)
               buffer = erase_transform(buffer)
               buffer = buffer.permute(1, 0, 2, 3)

          return buffer

     def loadvideo_decord(self, sample, sample_rate_scale=1, return_index=False):
          """Load video content using Decord"""
          fname = sample

          if not (os.path.exists(fname)):
               if return_index:
                    return [], []
               return []

          # avoid hanging issue
          if os.path.getsize(fname) < 1 * 1024:
               print('SKIP: ', fname, " - ", os.path.getsize(fname))
               if return_index:
                    return [], []
               return []
          try:
               if self.keep_aspect_ratio:
                    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
               else:
                    vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                   num_threads=1, ctx=cpu(0))
          except:
               print("video cannot be loaded by decord: ", fname)
               if return_index:
                    return [], []
               return []
          
          if self.mode == 'test':
               all_index = []
               tick = len(vr) / float(self.num_segment)
               all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                                   [int(tick * x) for x in range(self.num_segment)]))
               while len(all_index) < (self.num_segment * self.test_num_segment):
                    all_index.append(all_index[-1])
               all_index = list(np.sort(np.array(all_index))) 
               vr.seek(0)
               buffer = vr.get_batch(all_index).asnumpy()
               if return_index:
                    return buffer, np.array([idx/vr.get_avg_fps() for idx in all_index])
               return buffer

          # handle temporal segments
          average_duration = len(vr) // self.num_segment
          all_index = []
          if average_duration > 0:
               all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                         size=self.num_segment))
          elif len(vr) > self.num_segment:
               all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
          else:
               all_index += list(np.zeros((self.num_segment,)))
          all_index = list(np.array(all_index)) 
          vr.seek(0)
          buffer = vr.get_batch(all_index).asnumpy()
          if return_index:
               return buffer, np.array([idx/vr.get_avg_fps() for idx in all_index])
          return buffer

     def __len__(self):
          if self.mode != 'test':
               return len(self.dataset_samples)
          else:
               return len(self.test_dataset)
          
     def get_item_by_index(self, index):
        return self.__getitem__(index)
          

class HDEpicVideoClsDataset(EpicVideoClsDataset):
     def __init__(self, anno_path, data_path, mode='train', clip_len=8,
               crop_size=224, short_side_size=256, new_height=256,
               new_width=340, keep_aspect_ratio=True, num_segment=1,
               num_crop=1, test_num_segment=10, test_num_crop=3, args=None, audio_path=None):
          self.anno_path = anno_path
          self.data_path = data_path
          self.audio_path = args.audio_path
          self.mode = mode
          self.clip_len = clip_len
          self.crop_size = crop_size
          self.short_side_size = short_side_size
          self.new_height = new_height
          self.new_width = new_width
          self.keep_aspect_ratio = keep_aspect_ratio
          self.num_segment = num_segment
          self.test_num_segment = test_num_segment
          self.num_crop = num_crop
          self.test_num_crop = test_num_crop
          self.args = args
          self.aug = False
          self.rand_erase = False
          self.disable_video = args.disable_video
          if self.mode in ['train']:
               self.aug = True
               if self.args.reprob > 0:
                    self.rand_erase = True
          if VideoReader is None:
               raise ImportError("Unable to import `decord` which is required to read videos.")
          if self.audio_path is not None:
               self.audio_extension = '.wav'
               self.audio_type = args.audio_type
               self.realtime_audio = args.realtime_audio
               self.autosave_spec = args.autosave_spec
               self.data_set = 'EPIC_split'
               if (mode == 'train') and getattr(args, 'add_noise', None): 
                    self.spectrogram = Spectrogram(num_segment, args.audio_height, args.audio_width, n_fft=2048, process_type=args.process_type, noisereduce=args.noisereduce, specnorm=args.specnorm, noise=True)
               else:
                    self.spectrogram = Spectrogram(num_segment, args.audio_height, args.audio_width, n_fft=2048, process_type=args.process_type, noisereduce=args.noisereduce, specnorm=args.specnorm)
          
          import pandas as pd
          import pickle
          cleaned = pd.read_json(self.anno_path, lines=True)
          # remove empty main_action_classes
          cleaned = cleaned[cleaned['main_action_classes'].apply(lambda x: len(x) > 0 and len(x[0]) == 2)].reset_index(drop=True)
          # remove outliers
          cleaned = cleaned[(cleaned['main_action_classes'].apply(lambda x: x[0][0] < 97 and x[0][1] < 300))].reset_index(drop=True)
          
          # Check if all video files exist, print missing, and remove from dataset
          existing_samples = []
          missing_samples = []
          for sample_id in list(cleaned['unique_narration_id']):
               sample_path = os.path.join(self.data_path, sample_id + '.mp4')
               if not os.path.exists(sample_path):
                    print(f"Missing video file: {sample_path}")
                    missing_samples.append(sample_id)
               else:
                    existing_samples.append(sample_id)
          if missing_samples:
               print(f"Total missing videos: {len(missing_samples)} / {len(cleaned)}")
          cleaned = cleaned[cleaned['unique_narration_id'].isin(existing_samples)].reset_index(drop=True)
          # if self.mode == 'train':
          #      self.dataset_samples = list(cleaned.values[:, 0])[:101]
          # else:
          self.dataset_samples = list(cleaned['unique_narration_id'])
          verb_label_array = [item[0][0] for item in cleaned['main_action_classes']] # verb_idx
          noun_label_array = [item[0][1] for item in cleaned['main_action_classes']] # noun_idx
          action_label_array = [item[0][0]*300+item[0][1] for item in cleaned['main_action_classes']] # action
          if self.audio_path is not None:
               fps = 30
               self.audio_samples = {
               row.unique_narration_id: {
                    'start_frame': int(row.start_timestamp * fps),
                    'stop_frame': int(row.end_timestamp * fps)
               }
               for _, row in cleaned.iterrows()
               }
          self.narration_array = None
          self.label_array = np.stack((noun_label_array, verb_label_array, action_label_array), axis=1) # label [noun, verb] sequence
          
          if  (mode == 'train'):
               pass
          
          elif (mode == 'validation'):
               self.data_transform = video_transforms.Compose([
                    video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                    video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
               ])
          elif (mode == 'test'):
               self.data_resize = video_transforms.Compose([
                    video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
               ])
               self.data_transform = video_transforms.Compose([
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
               ])
               self.test_seg = []
               self.test_dataset = []
               self.test_label_array = []
               for ck in range(self.test_num_segment):
                    for cp in range(self.test_num_crop):
                         for idx in range(len(self.label_array)):
                              sample_label = self.label_array[idx]
                              self.test_label_array.append(sample_label)
                              self.test_dataset.append(self.dataset_samples[idx])
                              self.test_seg.append((ck, cp))



def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames

          
def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor