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
import torchaudio
import random

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
          self.audio_type = args.audio_type
          self.disable_video = False
          if self.mode in ['train']:
               self.aug = True
               if self.args.reprob > 0:
                    self.rand_erase = True
          if VideoReader is None:
               raise ImportError("Unable to import `decord` which is required to read videos.")
          if self.audio_path is not None:
               self._spectrogram_init(resampling_rate=24000)
          
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
          # self.audio_samples = pickle.load(open(audio_path, 'rb')) if audio_path is not None else None
          self.audio_samples = {cleaned.iloc[i, 0]: cleaned.iloc[i, 12:14] for i in range(len(cleaned))}
          # self.audio_samples = None
          # lavila_narrator = list(cleaned.values[:, 9])
          # self.lavila_narrator = [eval(nar) for nar in cleaned.values[:, 9]]
          del(cleaned)
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
          if self.mode == 'train':
               args = self.args
               scale_t = 1
               if self.audio_path is not None:
                    audio_trim_path = os.path.join(self.audio_path,'spec', self.audio_type, self.dataset_samples[index] + '.npy')
                    audio_trim_path = audio_trim_path.replace("single", "stacks") if self.audio_type == 'single' else audio_trim_path
                    if os.path.exists(audio_trim_path):
                         spec = self.loadaudiofromfile(audio_trim_path, self.audio_type)
                    else:
                         audio_id = '_'.join(self.dataset_samples[index].split('_')[:-1])
                         audio_sample = os.path.join(self.audio_path, 'wav', audio_id + '.wav')
                         start_frame = self.audio_samples[self.dataset_samples[index]]['start_frame']
                         end_frame = self.audio_samples[self.dataset_samples[index]]['stop_frame']
                         try:
                              spec = self.loadaudio(audio_sample, start_frame, end_frame, audio_type=self.audio_type, mode=self.mode)
                              if args.spec_augment:
                                   spec = self.spec_augment(spec)
                         except:
                              print("audio {} not correctly loaded during training, {}".format(audio_sample, self.dataset_samples[index]))
                              spec = torch.random((3, 16, 224, 224))
               else:
                    spec = {}
               
               sample = self.dataset_samples[index] + '.mp4'
               sample = os.path.join(self.data_path, sample)
               if self.disable_video:
                    return torch.tensor([1]), self.label_array[index], sample.split("/")[-1].split(".")[0], spec
               buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
               
               if len(buffer) == 0:
                    while len(buffer) == 0:
                         warnings.warn("video {} not correctly loaded during training".format(sample))
                         index = np.random.randint(self.__len__())
                         sample = self.dataset_samples[index]
                         buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
                         
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
               return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], spec
          
          elif self.mode == 'validation':
               if self.audio_path is not None:
                    audio_trim_path = os.path.join(self.audio_path,'spec', self.audio_type, self.dataset_samples[index] + '.npy')
                    audio_trim_path = audio_trim_path.replace("single", "stacks") if self.audio_type == 'single' else audio_trim_path
                    if os.path.exists(audio_trim_path):
                         spec = self.loadaudiofromfile(audio_trim_path, self.audio_type)
                    else:
                         audio_id = '_'.join(self.dataset_samples[index].split('_')[:-1])
                         audio_sample = os.path.join(self.audio_path, 'wav', audio_id + '.wav')
                         start_frame = self.audio_samples[self.dataset_samples[index]]['start_frame']
                         end_frame = self.audio_samples[self.dataset_samples[index]]['stop_frame']
                         spec = self.loadaudio(audio_sample, start_frame, end_frame, audio_type=self.audio_type)
               else:
                    spec = {}
               
               sample = self.dataset_samples[index] + '.mp4'
               sample = os.path.join(self.data_path, sample)
               if self.disable_video:
                    return torch.tensor([1]), self.label_array[index], sample.split("/")[-1].split(".")[0], spec
               buffer = self.loadvideo_decord(sample)
               
               if len(buffer) == 0:
                    while len(buffer) == 0:
                         warnings.warn("video {} not correctly loaded during validation".format(sample))
                         index = np.random.randint(self.__len__())
                         sample = self.dataset_samples[index]
                         buffer = self.loadvideo_decord(sample)
               buffer = self.data_transform(buffer)
               return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], spec
          
          elif self.mode == 'test':
               if self.audio_path is not None:
                    audio_trim_path = os.path.join(self.audio_path,'spec', self.audio_type, self.test_dataset[index] + '.npy')
                    audio_trim_path = audio_trim_path.replace("single", "stacks") if self.audio_type == 'single' else audio_trim_path
                    if os.path.exists(audio_trim_path):
                         spec = self.loadaudiofromfile(audio_trim_path, self.audio_type)
                    else:
                         audio_id = '_'.join(self.test_dataset[index].split('_')[:-1])
                         audio_sample = os.path.join(self.audio_path, 'wav', audio_id + '.wav')
                         start_frame = self.audio_samples[self.test_dataset[index]]['start_frame']
                         end_frame = self.audio_samples[self.test_dataset[index]]['stop_frame']
                         spec = self.loadaudio(audio_sample, start_frame, end_frame, audio_type=self.audio_type)
               else:
                    spec = {}
               
               sample = self.test_dataset[index] + '.mp4'
               sample = os.path.join(self.data_path, sample)
               chunk_nb, split_nb = self.test_seg[index]
               buffer = self.loadvideo_decord(sample)

               while len(buffer) == 0:
                    warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                    index = np.random.randint(self.__len__())
                    sample = self.test_dataset[index]
                    chunk_nb, split_nb = self.test_seg[index]
                    buffer = self.loadvideo_decord(sample)

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
               temporal_start = chunk_nb # 0/1
               spatial_start = int(split_nb * spatial_step)
               if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start::2, \
                         spatial_start:spatial_start + self.short_side_size, :, :]
               else:
                    buffer = buffer[temporal_start::2, \
                         :, spatial_start:spatial_start + self.short_side_size, :]

               buffer = self.data_transform(buffer)
               return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                    chunk_nb, split_nb, spec
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

     def add_noise(self, audio, noise_level=0.005):
          noise = torch.randn(audio.shape)
          return audio + noise_level * noise
     
     def _spectrogram_init(self, window_size=10,
                     step_size=5, resampling_rate=24000):
          nperseg = int(round(window_size * resampling_rate / 1e3))
          noverlap = int(round(step_size * resampling_rate / 1e3))

          # torchaudio를 사용한 스펙트로그램 계산
          # self.spectrogram = torch.nn.Sequential(
          #      torchaudio.transforms.Spectrogram(
          #           n_fft=447,
          #           win_length=nperseg,
          #           hop_length=noverlap,
          #           window_fn=torch.hann_window
          #      ),
          #      torchaudio.transforms.AmplitudeToDB()
          # )
          self.spectrogram = torch.nn.Sequential(
               torchaudio.transforms.MelSpectrogram(
                    sample_rate=resampling_rate,
                    n_fft=447,  # FFT 창 크기
                    win_length=nperseg,
                    hop_length=noverlap,
                    window_fn=torch.hann_window,
                    n_mels=224  # mel 스펙트로그램 빈의 수
               ),
               torchaudio.transforms.AmplitudeToDB()
          )
     
     # https://arxiv.org/abs/1904.08779
     def spec_augment(self, feat, T = 70, F = 20, time_mask_num = 2, freq_mask_num = 2):
          dim = feat.dim()
          feat_size = feat.size(-1)
          seq_len = feat.size(-2)

          # time mask
          for _ in range(time_mask_num):
               t = np.random.uniform(low=0.0, high=T)
               t = int(t)
               t0 = random.randint(0, seq_len - t)
               if dim == 2:
                    feat[t0 : t0 + t, :] = 0
               else:
                    feat[:, t0 : t0 + t, :] = 0

          # freq mask
          for _ in range(freq_mask_num):
               f = np.random.uniform(low=0.0, high=F)
               f = int(f)
               f0 = random.randint(0, feat_size - f)
               if dim == 2:
                    feat[:, f0 : f0 + f] = 0
               else:
                    feat[:, :, f0 : f0 + f] = 0

          return feat
          
     def _specgram(self, audio, window_size=10, step_size=5, eps=1e-6, resampling_rate=24000, target_length=1.119):
          current_length = audio.shape[-1] / resampling_rate
          if current_length != target_length:
               new_freq = int(round(resampling_rate * target_length / current_length/100) * 100)
               resample_transform = torchaudio.transforms.Resample(orig_freq=resampling_rate, new_freq=new_freq)
               audio = resample_transform(audio)
          spec = self.spectrogram(audio)
          if spec.shape[-1] != 224:
               expand = 224 - spec.shape[-1]
               if spec.dim() == 3:
                    spec = spec[:,:,:224] if spec.shape[-1] > 224 else torch.concat([spec,spec[:,:,-expand:]],dim=-1)
               else:
                    spec = spec[:,:224] if spec.shape[-1] > 224 else torch.concat([spec,spec[:,-expand:]],dim=-1)
          return spec
     
     def loadaudiofromfile(self, sample_path, audio_type='stack'):
          # audio_trim_path = os.path.join(self.audio_path,'spec', audio_type,)
          np_array = np.load(sample_path)
          spec = torch.tensor(np_array)
          if audio_type == 'stack':
               spec = spec.unsqueeze(0).unsqueeze(0).repeat(3, 16, 1, 1) if audio_type == 'stack' else spec.unsqueeze(0).repeat(3, 1, 1)
          elif audio_type == 'frame':
               spec = self.spectrogram(spec).unsqueeze(0).repeat(3,1,1,1)
          elif audio_type == 'all':
               spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
          elif audio_type == 'all8':
               spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
               spec = spec[:, [i for i in range(8) for _ in range(2)], :, :]
          elif audio_type in ['stacks','single']:
               spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
               stack_dim = spec.shape[1]
               if audio_type == 'stacks':
                    idx = np.round(np.linspace(0, stack_dim - 1, self.num_segment)).astype(int).tolist()
                    spec = spec[:, idx, :, :]
               else:
                    spec = spec[:, (stack_dim-1)//2, :, :]
          else:
               pass
          return spec
     
     def loadaudio(self, sample, start_frame, stop_frame, resampling_rate=24000, audio_type='stack', mode='test'):
          samples, sample_rate = torchaudio.load(sample)
          samples = samples.squeeze(0)
          left_sec = start_frame / 60
          right_sec = stop_frame / 60
          left_sample = int(round(left_sec * sample_rate))
          right_sample = int(round(right_sec * sample_rate))
          if right_sample > len(samples):
               right_sample = len(samples)
          length_sample = right_sample - left_sample
          length = int(round(1.119*sample_rate))
          if audio_type in ['stack','single']:
               stride = int(length_sample // length)
               if stride == 0:
                    if left_sample+length < len(samples):
                         samples = samples[left_sample:left_sample+length]
                    elif right_sample >= len(samples):
                         samples = samples[-length:]
                    else:
                         samples = samples[right_sample - length:right_sample]
               else:
                    samples = samples[left_sample:right_sample:stride]
                    samples = samples[:length]
               spec = self.spectrogram(samples)
               spec = spec.unsqueeze(0).unsqueeze(0).repeat(3, 16, 1, 1) if audio_type == 'stack' else spec.unsqueeze(0).repeat(3, 1, 1)
          elif audio_type == 'frame':
               average_duration = (stop_frame - start_frame) // self.num_segment
               all_index = []
               if average_duration > 0:
                    all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration, size=self.num_segment))
               else:
                    all_index = np.round(np.linspace(0, (stop_frame - start_frame), self.num_segment)).astype(int).tolist()
               spec = []
               for idx in all_index:
                    centre_sec = (start_frame  + idx) /60
                    left_sec = centre_sec - 0.559
                    right_sec = centre_sec + 0.559
                    left_sample = int(round(left_sec * sample_rate))
                    right_sample = int(round(right_sec * sample_rate))
                    length = int(round(1.118*sample_rate))
                    if left_sec < 0:
                         spec.append(samples[:length])
                    elif right_sample >= len(samples):
                         spec.append(samples[-length:])
                    else:
                         spec.append(samples[left_sample:right_sample])
               spec = torch.stack(spec, dim=0)
               spec = self.spectrogram(spec).unsqueeze(0).repeat(3,1,1,1)
          elif audio_type == 'all':
               if right_sample > len(samples):
                    right_sample = len(samples)
               step = int((right_sample-left_sample)//self.num_segment)
               samples = torch.stack([samples[i:i+step] for i in range(left_sample,right_sample,step)],dim=0)
               spec = self._specgram(samples, resampling_rate=sample_rate)
               spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
          elif audio_type == 'all8':
               step = step = int((right_sample-left_sample)//(self.num_segment/2))
               samples = torch.stack([samples[i:i+step] for i in range(left_sample,right_sample,step)],dim=0)
               spec = self._specgram(samples, resampling_rate=sample_rate)
               spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
               spec = spec[:, [i for i in range(8) for _ in range(2)], :, :]
          elif audio_type == 'stacks':
               stride = int(length_sample // length)
               if stride > 0:
                    samples = torch.stack([samples[left_sample+(i*length):left_sample+((i+1)*length)] for i in range(stride)],dim=0)
               else:
                    if left_sample+length < len(samples):
                         samples = samples[left_sample:left_sample+length]
                    elif right_sample >= len(samples):
                         samples = samples[-length:]
                    else:
                         samples = samples[right_sample - length:right_sample]
                    samples = samples.unsqueeze(0)
               spec = self.spectrogram(samples)
               spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
               stack_dim = spec.shape[1]
               if audio_type == 'stacks':
                    idx = np.round(np.linspace(0, stack_dim - 1, self.num_segment)).astype(int).tolist()
                    spec = spec[:, idx, :, :]
               else:
                    spec = spec[:, (stack_dim-1)//2, :, :]
          else:
               samples = samples[left_sample:right_sample]
               spec = self.spectrogram(samples)
               spec = spec.unsqueeze(0).unsqueeze(0).repeat(3, 16, 1, 1)
               # spec = spec.unsqueeze(0).unsqueeze(0).expand(3, 16, -1, -1)
          return spec

     def loadvideo_decord(self, sample, sample_rate_scale=1):
          """Load video content using Decord"""
          fname = sample

          if not (os.path.exists(fname)):
               return []

          # avoid hanging issue
          if os.path.getsize(fname) < 1 * 1024:
               print('SKIP: ', fname, " - ", os.path.getsize(fname))
               return []
          try:
               if self.keep_aspect_ratio:
                    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
               else:
                    vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                   num_threads=1, ctx=cpu(0))
          except:
               print("video cannot be loaded by decord: ", fname)
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
          return buffer

     def __len__(self):
          if self.mode != 'test':
               return len(self.dataset_samples)
          else:
               return len(self.test_dataset)
          

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