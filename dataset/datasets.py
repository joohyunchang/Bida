import os
from torchvision import transforms
from util_tools.transforms import *
from util_tools.masking_generator import TubeMaskingGenerator
from .kinetics import VideoClsDataset, VideoMAE
from .kinetics_sound import K400VidAudClsDataset
from .ucf101 import UCF101VidAudClsDataset
from .vggsound import VGGSoundVidAudClsDataset
from .ssv2 import SSVideoClsDataset
from .epic import EpicVideoClsDataset, HDEpicVideoClsDataset
from .epic_ov import EpicOVVideoClsDataset
from .epic_sounds import EpicSoundsVideoClsDataset
from .epic_dense import EpicDenseVideoClsDataset
from .diving import DivingVideoClsDataset
from .activitynet import ActivityNetDataset

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'kinetics400_train_gpt_xl.csv')
            anno_path = os.path.join(args.anno_path, 'kinetics400_train_audio.csv') if args.audio_path is not None else anno_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'kinetics400_test_gpt_xl.csv') 
            anno_path = os.path.join(args.anno_path, 'kinetics400_test_audio.csv') if args.audio_path is not None else anno_path
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'kinetics400_val_gpt_xl.csv') 
            anno_path = os.path.join(args.anno_path, 'kinetics400_val_audio.csv') if args.audio_path is not None else anno_path

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    elif args.data_set == 'Kinetics_sound':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'my_train.txt')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'my_test.txt')
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'my_test.txt')

        dataset = K400VidAudClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 31
    elif args.data_set == 'diving-48':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'train_gpt.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'val_gpt.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'val_gpt.csv')
    
        dataset = DivingVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 48
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'train_gpt_xl.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'val_gpt_xl.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'val_gpt_xl.csv')
    
        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174
    
    elif args.data_set =='MINI_SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'mini_train_mp4.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'mini_test_mp4.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'mini_val_mp4.csv')

        dataset = SSVideoClsDataset(
                anno_path=anno_path,
                data_path='/',
                mode=mode,
                clip_len=1,
                num_segment=args.num_frames,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                args=args)
        nb_classes = 87
    
    elif args.data_set == 'EPIC_OV':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'validation.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'validation.csv')

        dataset = EpicOVVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            audio_path=args.audio_path)
        nb_classes = 300
        
    elif args.data_set == 'EPIC':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'epic100_train_gpt2_xl.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'epic100_val_gpt2_xl.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'epic100_val_gpt2_xl.csv')

        dataset = EpicVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            audio_path=args.audio_path)
        nb_classes = 300
        
    elif args.data_set == 'HD_EPIC':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'HD_EPIC_Narrations.json')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'HD_EPIC_Narrations.json')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'HD_EPIC_Narrations.json')

        dataset = HDEpicVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            audio_path=args.audio_path)
        nb_classes = 300
        
    elif args.data_set == 'ActivityNet':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'val.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'val.csv')

        dataset = ActivityNetDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            audio_path=args.audio_path)
        nb_classes = 200
        
    elif args.data_set == 'EPIC_sounds':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'EPIC_Sounds_train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'EPIC_Sounds_validation.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'EPIC_Sounds_validation.csv')

        dataset = EpicSoundsVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            audio_path=args.audio_path)
        nb_classes = 44
        
    elif args.data_set == 'EPIC_dense':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'epic100_compo_train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'epic100_compo_val.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'epic100_compo_val.csv')

        dataset = EpicDenseVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 300
    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            if args.ucf101_type in ['1','2','3']:
                anno_path = os.path.join(args.anno_path, f'trainlist0{args.ucf101_type}_audio.txt')
            elif args.ucf101_type in ['all_1','all_2','all_3']:
                anno_path = os.path.join(args.anno_path, f'trainlist0{str.split(args.ucf101_type,"_")[-1]}.txt')
        elif test_mode is True:
            mode = 'test'
            if args.ucf101_type in ['1','2','3']:
                anno_path = os.path.join(args.anno_path, f'testlist0{args.ucf101_type}_audio.txt')
            elif args.ucf101_type in ['all_1','all_2','all_3']:
                anno_path = os.path.join(args.anno_path, f'testlist0{str.split(args.ucf101_type,"_")[-1]}_label.txt')
        else:  
            mode = 'validation'
            if args.ucf101_type in ['1','2','3']:
                anno_path = os.path.join(args.anno_path, f'testlist0{args.ucf101_type}_audio.txt')
            elif args.ucf101_type in ['all_1','all_2','all_3']:
                anno_path = os.path.join(args.anno_path, f'testlist0{str.split(args.ucf101_type,"_")[-1]}_label.txt')

        dataset = UCF101VidAudClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
        
    elif args.data_set == 'VGGSound':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.anno_path, 'train_sel.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.anno_path, 'test_sel.csv')
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.anno_path, 'test_sel.csv')

        dataset = VGGSoundVidAudClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 309
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    
    # assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes