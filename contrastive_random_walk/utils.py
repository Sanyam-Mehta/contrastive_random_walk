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
