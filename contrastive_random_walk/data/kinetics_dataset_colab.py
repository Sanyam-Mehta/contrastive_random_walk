import time

import numpy as np

from contrastive_random_walk.utils import extract_patches_with_jitter, make_palindrome
from torchvision import torch
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.kinetics import Kinetics
from torchvision.datasets.utils import list_dir

from torchvision.datasets.video_utils import VideoClips
from torchvision import transforms as T

tranformations_final = T.Compose(
    [
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

class KineticsCustom():
    def __init__(
        self,
        root,
        frames_per_clip,
        split="train",
        step_between_clips=1,
        frame_rate=None,
        extensions=("mp4",),
        _precomputed_metadata=None,
        num_classes=400,
        transform_video=None,
        tranformations_frame=None,
        return_palindrome=False,
    ):
        self.root = root
        self.classes = list(sorted(list_dir(root)))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(
            self.root, self.class_to_idx, extensions, is_valid_file=None
        )
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
        )
        self.transform_video = transform_video
        self.tranformations_frame = tranformations_frame
        self.return_palindrome = return_palindrome

    def __getitem__(self, idx):
        # Get the video from the index
        # print("Getting video from index")
        video, video_path = self.get_video_from_index(idx)
        # print("Got video from index")
        # print("Video Shape: ", video.shape)
        # video shape: (T, H, W, C) and channels dimension is last


        # print("Shape of video: ", video.shape)
        # original_video = video.unsqueeze(1)

        # Transform each frame as follows:
        # 1. Convert the frame to a 256*256 image
        # 2. Convert each frame to a 7*7 grid of 64*64 patches using extract_patches_with_jitter function
        # 3. Apply the transform on each patch

        # The transformed video should have the shape (T, N, H, W, C)
        # where N is the number of patches (49 in this case)
        # H is the height of each patch (64)
        # W is the width of each patch (64)
        # C is the number of channels in the input tensor (3)

        # Actual code begins here:
        # print("Starting patch extraction")
        new_video = []
        video_patches = []

        # TODO: Poorly written code. Should be refactored
        for i in range(video.shape[0]):
            # img shape is (H, W, C). It is a tensor of shape (64, 64, 3)
            img = video[i].permute(2, 0, 1)
            # print("Image Shape: ", img.shape)
            # print("Transforming Image to 256*256")
            # print("Image Type: ", type(img))
            img = self.transform_video(img)
            # print("Transformed Image Shape: ", img.shape)
            img = img.permute(1, 2, 0)
            _, modified_patches = extract_patches_with_jitter(
                img,
                transforms=self.tranformations_frame,
            )
            video_patches.append(modified_patches)
            new_video.append(tranformations_final(img.permute(2, 0, 1)).permute(1, 2, 0))

       
        # print("Patch extraction done")
        # # transform the list into a palindrome:
        if self.return_palindrome:
            video_patches = make_palindrome(video_patches)

        video_patches = np.stack(video_patches)

        video_patches = torch.tensor(video_patches)

        # print(video_patches.shape)
        video_patches = video_patches.permute(0, 1, 3, 4, 2)

        # print(video_patches.shape)
        # print("Above Is video patches ka shape")

        new_video = torch.stack(tuple(new_video))

        # video shape: (T, H, W, C) and channels dimension is last
        video = new_video.unsqueeze(1)  # T, NxN, H, W, C where N == 1

        # video_patches has dimensions (2*clip_len/clip_len, 49, 64, 64, 3) [2*T, NxN, H, W, C]
        return {
            "video_patches": video_patches,
            "video": video,
            "dataset_idx": idx,
            "video_path": video_path,
        }

    def __len__(self) -> int:
      return self.video_clips.num_clips()


    def get_video_from_index(self, idx):
        video, _, _, video_idx = self.video_clips.get_clip(idx)

        video_path =  self.video_clips.video_paths[video_idx] 

        # Inside Video Transform
        # video = self.transform_video(video)

        # Transform the vide frame by frame
        # print("Transforming video frame by frame")
        # for i in range(video.shape[0]):
        #     video[i] = self.transform_video(video[i])

        # video shape: (T, H, W, C) and channels dimension is last
        assert video.shape[3] == 3, "Video should have 3 channels"
        return video, video_path  # video, audio, self.class_to_idx[info["label"]]


class KineticsCustomTest:
    def __init__(
        self,
        dummy_video,
        transform_video=None,
        tranformations_frame=None,
        return_palindrome=False,
    ):
        self.dummy_video = dummy_video
        self.transform_video = transform_video
        self.tranformations_frame = tranformations_frame
        self.return_palindrome = return_palindrome

    def get_video_from_index(self, idx):
        dummy_video = self.transform_video(self.dummy_video)
        return dummy_video

    def __getitem__(self, idx):
        video = self.get_video_from_index(idx)
        print("dummy video transofmration done")

        # Direct copy from KineticsCustom.__getitem__ method
        # Transform each frame as follows:
        # 1. Convert the frame to a 256*256 image
        # 2. Convert each frame to a 7*7 grid of 64*64 patches using extract_patches_with_jitter function
        # 3. Apply the transform on each patch

        # The transformed video should have the shape (T, N, H, W, C)
        # where N is the number of patches (49 in this case)
        # H is the height of each patch (64)
        # W is the width of each patch (64)
        # C is the number of channels in the input tensor (3)

        # Actual code begins here:
        time_start = time.time()
        video_patches = []
        for i in range(video.shape[0]):
            # img shape is (H, W, C). It is a tensor of shape (64, 64, 3)
            img = video[i]
            _, modified_patches = extract_patches_with_jitter(
                img,
                transforms=self.tranformations_frame,
            )
            video_patches.append(modified_patches)

        time_end = time.time()

        # print time taken in seconds for patch extraction
        print("Time taken for patch extraction in seconds: ", time_end - time_start)

        # # transform the list into a palindrome:
        if self.return_palindrome:
            video_patches = make_palindrome(video_patches)

        video_patches = np.stack(video_patches)

        # video shape: (T, H, W, C) and channels dimension is last
        video = video.unsqueeze(1)  # T, NxN, H, W, C where N == 1

        # video_patches has dimensions (2*clip_len, 49, 64, 64, 3) [2*T, NxN, H, W, C]
        return video_patches, video
