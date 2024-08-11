import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.kinetics import Kinetics

from contrastive_random_walk.utils import extract_patches_with_jitter

import numpy as np

class KineticsCustom(Kinetics):
    def __init__(
        self,
        root,
        frames_per_clip,
        step_between_clips=1,
        frame_rate=None,
        extensions=("mp4",),
        transform=None,
        cached=None,
        _precomputed_metadata=None,
        num_classes=400,
    ):
        super(KineticsCustom, self).__init__(
            root,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            extensions,
            transform,
            cached,
            _precomputed_metadata,
            num_classes=num_classes,
        )
        self.classes = list(sorted(list_dir(root)))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(self.root, self.class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
        )
        self.transform = transform

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)

        # Video shape: (T, C, H, W) 
        video = video.permute(0, 3, 1, 2)

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
        video_patches = []
        for i in range(video.shape[0]):
            img = video[i]
            _, modified_patches = extract_patches_with_jitter(
                img,
                transforms=self.transform,
            )
            video_patches.append(modified_patches)

        video_patches = np.stack(video_patches)
        return video_patches # video, audio, self.class_to_idx[info["label"]]