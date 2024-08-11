import numpy as np
from contrastive_random_walk.data.kinetics_dataset import KineticsCustomTest
import torchvision.transforms as T
import torch

def test_kinetics_custom_transformations():

    # Define dummy transformations
    transforms_video = T.Compose(
        [
            T.Resize(256),
            T.ColorJitter(),
        ]
    )

    tranformations_frame = T.Compose(
        [
            # RandomResizedCrop is the spatial jitter transofrmation
            T.RandomResizedCrop((64, 64), scale=(0.7, 0.9), ratio=(0.7, 1.3)),
        ]
    )

    # Create a dummy video (e.g., 10 frames of 256x256 RGB images)
    num_frames = 10
    height, width, channels = 448, 448, 3
    dummy_video = np.random.randint(0, 256, (num_frames, height, width, channels), dtype=np.uint8)

    # dummy_video has the shape (10, 1024, 956, 3)

    # Convert dummy video to a torch tensor
    dummy_video_tensor = torch.from_numpy(dummy_video)

    # Instantiate KineticsCustom with dummy transformations
    print("Starting test case 1")
    kinetics_custom_dataset = KineticsCustomTest(
        dummy_video=dummy_video_tensor,
        transform_video=transforms_video,
        tranformations_frame=tranformations_frame
    )

    # Apply transformations to the dummy video
    video_patches = kinetics_custom_dataset.__getitem__(1)

    print(video_patches.shape)

    # Check the output dimensions
    
    print("All test cases passed (KineticsCustomTest)!")

# Run the test
test_kinetics_custom_transformations()