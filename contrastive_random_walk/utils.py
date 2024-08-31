import numpy as np


def patchify_image(image, patch_size):
    """
    Extracts patches from an image based on the given patch size.
    Args:
        image (numpy.ndarray): The input image.
        patch_size (tuple): The size of each patch in (height, width) format.
    Returns:
        numpy.ndarray: An array of patches extracted from the image.
    """
    # Get the height and width of the image
    # Get the height and width of each patch
    # Initialize an empty list to store the patches
    # Assert that the patch sizes are valid
    # Iterate over the image in patch-sized steps
    # Extract the patch from the image
    # Append the patch to the list of patches
    # Convert the list of patches to a numpy array

    # Check that image is not empty
    assert image.size > 0, "Empty image"

    height, width = image.shape[:2]
    patch_height, patch_width = patch_size

    assert height % patch_height == 0 and width % patch_width == 0, "Invalid patch size"

    assert patch_height > 0 and patch_width > 0, "Invalid patch size"

    patches = []
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patch = image[i : i + patch_height, j : j + patch_width]
            patches.append(patch)

    return np.array(patches)


def make_palindrome(lst):
    """
    Converts a list into a palindrome by appending the reverse of the list to itself.
    Args:
        lst (list): The input list.
    Returns:
        list: The palindrome list.
    """
    # Create a copy of the input list
    # Reverse the copy of the list
    # Append the reversed list to the original list
    # Return the palindrome list

    palindrome = lst.copy()
    lst.reverse()
    palindrome.extend(lst)

    return palindrome


def extract_patches_with_jitter(image, transforms=None):
    """
    Extracts 64x64 patches from a 256x256 image on a 7x7 grid with overlapping patches,
    and combines them into a single 448x448 image.

    Args:
        image (np.ndarray): A 256x256 image.
        jitter (int): The maximum amount of jitter to apply to the patch coordinates.

    Returns:
        np.ndarray: A 448x448 image containing the extracted patches.
        np.ndarray: An array of extracted patches.

    P.S.: Spatial jittering in the original paper is a RandomResizedCrop operation
    """
    image = image.detach().cpu().numpy()
    patch_size = 64
    stride = 32

    patches = []

    new_image_size = 448
    new_image = (
        np.zeros((new_image_size, new_image_size, image.shape[2]), dtype=image.dtype)
        if len(image.shape) == 3
        else np.zeros((new_image_size, new_image_size), dtype=image.dtype)
    )

    for i in range(7):
        for j in range(7):
            x_start = i * stride
            y_start = j * stride
            patch = image[
                x_start : x_start + patch_size, y_start : y_start + patch_size
            ]

            # Implement jittering here (RandomResizedCrop, use A.RandomResizedCrop)
            augmented = transforms(patch)
            jittered_patch = augmented

            new_x_start = i * patch_size
            new_y_start = j * patch_size
            new_image[
                new_x_start : new_x_start + patch_size,
                new_y_start : new_y_start + patch_size,
            ] = jittered_patch

            patches.append(jittered_patch)

    # patches is a list of 49 patches, each of size 64x64

    return new_image, np.array(patches)
