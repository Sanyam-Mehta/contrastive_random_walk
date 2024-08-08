import numpy as np
from contrastive_random_walk.utils import make_palindrome, patchify_image


def test_make_palindrome():
    # Test case 1: Empty list
    lst = []
    expected_output = []
    assert make_palindrome(lst) == expected_output

    # Test case 2: List with one element
    lst = [1]
    expected_output = [1, 1]
    assert make_palindrome(lst) == expected_output

    # Test case 3: List with even number of elements
    lst = [1, 2, 3, 4]
    expected_output = [1, 2, 3, 4, 4, 3, 2, 1]
    assert make_palindrome(lst) == expected_output

    # Test case 4: List with odd number of elements
    lst = [1, 2, 3, 4, 5]
    expected_output = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
    assert make_palindrome(lst) == expected_output

    # Test case 5: List with duplicate elements
    lst = [1, 1, 2, 2, 3, 3]
    expected_output = [1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1]
    assert make_palindrome(lst) == expected_output

    print("All test cases passed (Palindrome)!")


test_make_palindrome()


def test_patchify_image():
    # Test case 1: Empty image
    image = np.array([])
    patch_size = (2, 2)
    # The function should raise an AssertionError for an empty image
    try:
        patchify_image(image, patch_size)
    except AssertionError:
        pass
    else:
        assert False

    # Test case 2: Image with dimensions smaller than patch size
    image = np.array([[1, 2], [3, 4]])
    patch_size = (3, 3)
    # The function should raise an AssertionError for an invalid patch size
    try:
        patchify_image(image, patch_size)
    except AssertionError:
        pass
    else:
        assert False

    # Test case 3: Image with dimensions divisible by patch size
    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    patch_size = (2, 2)
    expected_output = np.array(
        [[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]]
    )
    assert np.array_equal(patchify_image(image, patch_size), expected_output)

    # Test case 4: Image with dimensions not divisible by patch size
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    patch_size = (2, 2)
    # The function should raise an AssertionError for an invalid patch size
    try:
        patchify_image(image, patch_size)
    except AssertionError:
        pass
    else:
        assert False

    # Test case 5: Image with dimensions (6 * 18) and patch size (3 * 9)
    image = np.random.randint(0, 255, (6, 18))
    patch_size = (3, 9)
    # The function should return an array of patches with shape (4, 3, 9)
    assert patchify_image(image, patch_size).shape == (4, 3, 9)

    print("All test cases passed (Patchify)!")


test_patchify_image()
