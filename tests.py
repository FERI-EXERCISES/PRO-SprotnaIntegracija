import unittest
from main import *


class TestBinarySearch(unittest.TestCase):
    def test_find_middle_element(self):
        result = binary_search([1, 2, 3, 4, 5], 3)
        self.assertEqual(result, 2, "Should find element at index 2")

    def test_element_not_found(self):
        result = binary_search([1, 2, 3, 4, 5], 6)
        self.assertEqual(result, -1, "Should return -1 for not found")

    def test_large_list_performance(self):
        large_list = list(range(1000000))  # seznam z 1.000.000 elementi
        target = 999999  # element na koncu seznama
        result = binary_search(large_list, target)
        self.assertEqual(result, target, "Should find element at the end of a large list")


class TestTemperatureConversion(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        result = convert_temperature(100, 'fahrenheit')
        self.assertEqual(result, 212, "Should convert 100C to 212F")

    def test_fahrenheit_to_celsius(self):
        result = convert_temperature(212, 'celsius')
        self.assertEqual(result, 100, "Should convert 212F to 100C")

    def test_negative_temperature(self):
        result = convert_temperature(-15, 'fahrenheit')
        self.assertEqual(result, 5, "Should convert -15C to 5F")

    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
            convert_temperature(100, 'kelvin')


class TestBubbleSort(unittest.TestCase):
    def test_sorting(self):
        result = bubble_sort([3, 2, 1, 5, 4])
        self.assertEqual(result, [1, 2, 3, 4, 5], "Should sort the list")

    def test_already_sorted(self):
        result = bubble_sort([1, 2, 3, 4, 5])
        self.assertEqual(result, [1, 2, 3, 4, 5], "Should work with already sorted list")

    def test_empty_list(self):
        result = bubble_sort([])
        self.assertEqual(result, [], "Should work with empty list")

    def test_single_element_list(self):
        result = bubble_sort([1])
        self.assertEqual(result, [1], "Should work with single element list")


class TestConv2d(unittest.TestCase):
    def TestConv2d(self):
        slika = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        jedro = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        result = conv_2d(slika, jedro)
        self.assertEqual(result, [[[-6, 0, 6], [-6, 0, 6], [-6, 0, 6]]], "Should perform 2d convolution")

    def test_invalid_image_size(self):
        slika = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        jedro = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        with self.assertRaises(ValueError):
            conv_2d(slika, jedro)

    def test_invalid_kernel_size(self):
        slika = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        jedro = np.array([[1, 0, -1], [1, 0, -1]], dtype=np.float32)
        with self.assertRaises(ValueError):
            conv_2d(slika, jedro)


class TestOpen(unittest.TestCase):
    def test_output_bool(self):
        slika = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        jedro = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        result = open(slika, jedro)
        self.assertEqual(result.dtype, bool, "Should return a boolean array")

    def test_input_not_bool(self):
        slika = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        jedro = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            open(slika, jedro)

    def test_invalid_input(self):
        with self.assertRaises(RuntimeError):
            open(None, None)


class TestPrevzorciSlikoFFT(unittest.TestCase):
    def test_output_shape(self):
        slika = np.random.rand(100, 100, 3)
        nova_visina, nova_sirina = 200, 200
        result = prevzorci_sliko_fft(slika, nova_visina, nova_sirina)
        self.assertEqual(result.shape, (nova_visina, nova_sirina, 3), "Should return an image with the correct shape")

    def test_invalid_input(self):
        with self.assertRaises(AttributeError):
            prevzorci_sliko_fft(None, 100, 100)

    def test_invalid_output_shape(self):
        slika = np.random.rand(100, 100, 3)
        with self.assertRaises(ValueError):
            prevzorci_sliko_fft(slika, 0, 0)


class TestRGB_v_YCbCr(unittest.TestCase):
    def test_output_shape(self):
        slika = np.random.rand(100, 100, 3)
        result = RGB_v_YCbCr(slika)
        self.assertEqual(result.shape, (100, 100, 3), "Should return an image with the correct shape")

    def test_invalid_input(self):
        slika = np.random.rand(100, 100)
        with self.assertRaises(ValueError):
            RGB_v_YCbCr(slika)

    def test_nonstandard_input_datatype(self):
        slika = np.random.rand(100, 100, 3).astype(np.float32)
        result = RGB_v_YCbCr(slika)
        self.assertEqual(result.dtype, np.uint8, "Should return an image of type uint8")
        self.assertEqual(result.shape, (100, 100, 3), "Should return an image with the correct shape")


class TestKonvolucijaFFT(unittest.TestCase):
    def test_output_shape(self):
        signal = np.random.rand(100)
        impulz = np.array([0.5, 1, 0.5])
        rob = 'ničle'
        result = konvolucija_fft(signal, impulz, rob)
        self.assertEqual(result.shape, (100), "Should return an image with the correct shape")

    def test_invalid_input(self):
        signal = np.random.rand(100, 100)
        impulz = np.random.rand(5, 5)
        rob = 'ničle'
        with self.assertRaises(ValueError):
            konvolucija_fft(signal, impulz, rob)

    def test_invalid_edge_handling(self):
        signal = np.random.rand(100, 100)
        impulz = np.random.rand(5)
        rob = 'invalid'
        with self.assertRaises(UnboundLocalError):
            konvolucija_fft(signal, impulz, rob)


class TestIsPrime(unittest.TestCase):
    def test_prime_number(self):
        result = is_prime(7)
        self.assertTrue(result, "Should return True for prime number")

    def test_non_prime_number(self):
        result = is_prime(9)
        self.assertFalse(result, "Should return False for non-prime number")

    def test_negative_number(self):
        result = is_prime(-7)
        self.assertFalse(result, "Should return False for negative number")

    def test_zero(self):
        result = is_prime(0)
        self.assertFalse(result, "Should return False for zero")


class TestFindMusicFiles(unittest.TestCase):
    def test_output_type(self):
        result = find_music_files('C:/Users/simon/Music/Test', ['mp3', 'flac', 'wav'])
        self.assertIsInstance(result, list, "Should return a list")

    def test_invalid_start_path(self):
        with self.assertRaises(ValueError):
            find_music_files('invalid_path', ['mp3', 'flac', 'wav'])

    def test_invalid_extensions(self):
        with self.assertRaises(ValueError):
            find_music_files('C:/Users/Anja/PycharmProjects/PythonProject', ['invalid'])


if __name__ == '__main__':
    unittest.main()
