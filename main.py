import fnmatch
from scipy.ndimage import binary_erosion, binary_dilation
import os
import tests
import numpy as np


# function 1 (GPT)
def binary_search(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return -1


# function 2
def convert_temperature(temp, to_scale):
    if to_scale.lower() == 'fahrenheit':
        return temp * 9 / 5 + 32
    elif to_scale.lower() == 'celsius':
        return (temp - 32) * 5 / 9
    else:
        raise ValueError("Unsupported scale")


# function 3 (GPT)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# function 4
def conv_2d(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    # Dimenzije
    H, W, _ = slika.shape
    N, M = jedro.shape

    # Izračun središče jedra
    C_N, C_M = N // 2, M // 2

    # Priprava vhodne slike
    tmp = np.pad(slika, ((C_N, C_M), (C_N, C_M), (0, 0)), mode='constant')

    # Pripravljenje izhodne slike
    izhod = np.zeros((H, W, 3), dtype=np.float32)

    # Izvedi konvolucijo
    for i in range(H):
        for j in range(W):
            izhod[i, j] = (tmp[i:i + N, j:j + M] * jedro).sum(axis=(0, 1))

    return izhod


# function 5
def open(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(slika, jedro)
    opened = binary_dilation(eroded, jedro)
    return opened.astype(bool)


# function 6
def fft2_resample(x: np.ndarray, new_shape: tuple) -> np.ndarray:
    # Izračunamo 2D FFT in izvedemo premik frekvence 0 v središče
    fft = np.fft.fftshift(np.fft.fft2(x))
    fft_resampled = np.zeros(new_shape, dtype=complex)

    # Izračunamo minimalne dimenzije za obrezovanje ali dodajanje frekvenc
    min_x = min(x.shape[0], new_shape[0]) // 2
    min_y = min(x.shape[1], new_shape[1]) // 2

    # Obrezovanje ali dodajanje frekvenc, da dosežemo želeno velikost
    fft_resampled[new_shape[0] // 2 - min_x: new_shape[0] // 2 + min_x,
    new_shape[1] // 2 - min_y: new_shape[1] // 2 + min_y] = \
        fft[x.shape[0] // 2 - min_x: x.shape[0] // 2 + min_x, x.shape[1] // 2 - min_y: x.shape[1] // 2 + min_y]

    return np.fft.ifft2(np.fft.ifftshift(fft_resampled))


def prevzorci_sliko_fft(slika: np.ndarray, nova_visina: int, nova_sirina: int) -> np.ndarray:
    # Preveri, ali je slika barvna ali sivinska
    if slika.ndim == 3:
        # Izvajamo vzorčenje Fouriera za vsak barvni kanal posebej in shranjujemo rezultate v ustrezni kanal rezultatne slike
        rezultat = np.empty((nova_visina, nova_sirina, 3), dtype=slika.dtype)

        for i in range(3):
            rezultat[..., i] = np.abs(fft2_resample(slika[..., i], (nova_visina, nova_sirina)))
    else:
        # Če je slika enobarvna izvajamo vzorčenje Fouriera neposredno na sliki
        rezultat = np.abs(fft2_resample(slika, (nova_visina, nova_sirina)))

    return rezultat


# function 7
def RGB_v_YCbCr(slika: np.ndarray) -> np.ndarray:
    # Definiraj koeficiente za pretvorbo v YCbCr
    YCbCr_from_RGB = np.array([[0.299, 0.587, 0.114],
                               [-0.168736, -0.331264, 0.5],
                               [0.5, -0.418688, -0.081312]])

    YCbCr = np.dot(slika, YCbCr_from_RGB.T)
    YCbCr[:, :, [1, 2]] += 128
    return np.uint8(YCbCr)


# function 8
def konvolucija_fft(signal: np.ndarray, impulz: np.ndarray, rob: str) -> np.ndarray:
    N = len(signal)
    K = len(impulz)

    impulz_padded = np.pad(impulz, (0, N - K), mode='constant', constant_values=0)

    # Padding the signal and impulse response according to the specified edge handling
    if rob == 'ničle':
        signal_padded = np.pad(signal, (0, K - 1), mode='constant', constant_values=0)
        impulz_padded = np.pad(impulz, (0, N - 1), mode='constant', constant_values=0)
    elif rob == 'zrcaljen':
        signal_padded = np.pad(signal, (K - 1, K - 1), mode='reflect')
        impulz_padded = np.pad(impulz, (N - 1, N - 1), mode='reflect')
    elif rob == 'krožni':
        signal_padded = np.pad(signal, (K - 1, K - 1), mode='wrap')
        impulz_padded = np.pad(impulz, (N - 1, N - 1), mode='wrap')

    # Calculating the convolution in frequency domain
    signal_fft = np.fft.fft(signal_padded)
    impulz_fft = np.fft.fft(impulz_padded)
    output_fft = signal_fft * impulz_fft
    output = np.fft.ifft(output_fft)[:N].real

    # Reshaping the output to match the dimensions of the input signal
    if signal.ndim > 1:
        output = output.reshape((N, signal.shape[1]))

    return output


# function 9 (GPT)
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


# function 10
def find_music_files(start_path, extensions):
    music_files = []

    for root, dirs, files in os.walk(start_path):
        for extension in extensions:
            for filename in fnmatch.filter(files, f'*.{extension}'):
                music_files.append(os.path.join(root, filename))
                artistname = filename.split(" - ")[0]
                songname = os.path.splitext("".join(filename.split(" - ")[1:]))[0]
                path = os.path.join(root, filename)
                filesize = os.path.getsize(os.path.join(root, filename))
                fileextension = filename.split(".")[-1]

    return music_files


if __name__ == '__main__':
    print("Hello, World")
