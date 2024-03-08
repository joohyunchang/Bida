import torch
import torchaudio
import numpy as np
import random

class Spectrogram:
    def __init__(self, num_segment=16, n_mels=224, length=224, window_size=10, step_size=5, n_fft=447, resampling_rate=24000):
        nperseg = int(round(window_size * resampling_rate / 1e3))
        noverlap = int(round(step_size * resampling_rate / 1e3))
        self.num_segment = num_segment
        self.length = length
        self.sec = length * 0.004995535714

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
                n_fft=n_fft,  # FFT 창 크기
                win_length=nperseg,
                hop_length=noverlap,
                window_fn=torch.hann_window,
                n_mels=n_mels  # mel 스펙트로그램 빈의 수
            ),
            torchaudio.transforms.AmplitudeToDB()
        )
        
    def add_noise(self, audio, noise_level=0.005):
        noise = torch.randn(audio.shape)
        return audio + noise_level * noise
        
    def _specgram(self, audio, window_size=10, step_size=5, eps=1e-6, resampling_rate=24000, target_length=1.119):
        # current_length = audio.shape[-1] / resampling_rate
        # if current_length != target_length:
        #     new_freq = int(round(resampling_rate * target_length / current_length/100) * 100)
        #     resample_transform = torchaudio.transforms.Resample(orig_freq=resampling_rate, new_freq=new_freq)
        #     audio = resample_transform(audio)
        spec = self.spectrogram(audio)
        if spec.shape[-1] != self.length:
            expand = self.length - spec.shape[-1]
            if spec.dim() == 3:
                spec = spec[:,:,:self.length] if spec.shape[-1] > self.length else torch.concat([spec,spec[:,:,-expand:]],dim=-1)
            else:
                spec = spec[:,:self.length] if spec.shape[-1] > self.length else torch.concat([spec,spec[:,-expand:]],dim=-1)
        return spec

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

    def loadaudiofromfile(self, sample_path, audio_type='stack'):
        # audio_trim_path = os.path.join(self.audio_path,'spec', audio_type,)
        np_array = np.load(sample_path)
        spec = torch.tensor(np_array)
        if audio_type == 'stack':
            spec = spec.unsqueeze(0).unsqueeze(0).repeat(3, 16, 1, 1) if audio_type == 'stack' else spec.unsqueeze(0).repeat(3, 1, 1)
        elif audio_type == 'frame':
            spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
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
        elif audio_type in ['onespec', 'single1024']:
            spec = spec.unsqueeze(0).repeat(3, 1, 1)
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
        length = int(round(self.sec*sample_rate))
        if audio_type in ['stack']:
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
                left_sec = centre_sec - self.sec/2
                right_sec = centre_sec + self.sec/2
                left_sample = int(round(left_sec * sample_rate))
                right_sample = int(round(right_sec * sample_rate))
                length = int(round(self.sec*sample_rate))
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
            spec = self._specgram(samples, resampling_rate=sample_rate, target_length=self.sec)
            spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
        elif audio_type == 'all8':
            step = step = int((right_sample-left_sample)//(self.num_segment/2))
            samples = torch.stack([samples[i:i+step] for i in range(left_sample,right_sample,step)],dim=0)
            spec = self._specgram(samples, resampling_rate=sample_rate, target_length=self.sec)
            spec = spec.unsqueeze(0).repeat(3, 1, 1, 1)
            spec = spec[:, [i for i in range(8) for _ in range(2)], :, :]
        elif audio_type in ['stacks','single','single1024']:
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
        elif audio_type == 'onespec':
            samples = samples[left_sample:right_sample]
            spec = self._specgram(samples, resampling_rate=sample_rate, target_length=self.sec)
            spec = spec.unsqueeze(0).repeat(3, 1, 1)
        else:
            samples = samples[left_sample:right_sample]
            spec = self.spectrogram(samples)
            spec = spec.unsqueeze(0).unsqueeze(0).repeat(3, 16, 1, 1)
            # spec = spec.unsqueeze(0).unsqueeze(0).expand(3, 16, -1, -1)
        return spec