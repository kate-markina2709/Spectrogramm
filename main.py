import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

""" Кратковременное преобразование Фурье (STFT) """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # Нули в начале (таким образом, центр 1-го окна должен быть для образца nr. 0)
    # np.floor - округление вниз до ближайшего целого
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # столбцы для отображения окон
    # np.ceil - округление вверх до ближайшего целого
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # нули в конце (таким образом, образцы могут быть полностью покрыты рамками)
    samples = np.append(samples, np.zeros(frameSize))
    # samples.strides[0] - шаг
    # stride_tricks.as_strided - представление массива с указанной формой и смещением байтов между элементами для перехода по ним вдоль разных осей
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    # функция вычисляет одномерное n-точечное дискретное преобразование Фурье (DFT) массива вещественных значений
    # с помощью эффективного алгоритма, называемого быстрым преобразованием Фурье (FFT)
    return np.fft.rfft(frames)

""" логарифмическое масштабирование частотнуй оси """
def logscale_spec(spec, sr=44100, factor=20.):
    # timebins - временные рамки; freqbins - частотные диапазоны
    timebins, freqbins = np.shape(spec)
    # np.linspace - возвращает одномерный массив из указанного количества элементов,
    # значения которых равномерно распределенны внутри заданного интервала
    # scale - масштаб
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    # np.unique - находит уникальные элементы массива и возвращает их в отсортированном массиве
    scale = np.unique(np.round(scale))
    # создайте спектрограмму с новыми ячейками частоты
    # np.complex128 - комплексные числа в которых действительная и мнимая части представлены двумя вещественными числами типа float64
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            # : - срез; :i - скопировать с начала по i эл-т; i: - скопировать с i эл-та и до конца
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i+1])], axis=1)
    # список центральной частоты ячеек
    #  np.fft.fftfreq - Возвращаемый массив с плавающей точкой f содержит центры ячеек частоты в циклах на единицу интервала выборки (с нулем в начале).
    # Например, если интервал дискретизации равен секундам, то единица измерения частоты равна циклам в секунду.
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            # np.mean - вычисляет среднее арифметическое значений элементов массива
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" построение спектрограммы"""
# colormap="jet" - было
def plotstft(num, audiopath, binsize=2**10, plotpath=None, colormap="viridis"):
    #samplerate - частота; samples - образцы (их число совпадает с частотой)
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    # амплитуда в децибелах
    ims = 20.*np.log10(np.abs(sshow)/10e-6)

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    # Добавить цветовую полосу на график
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    # plt.xticks - Получите или установите текущие местоположения галочек и метки по оси x
    # %.02f - точность
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    # Целые числа в диапазоне от -32768 по 32767, (числа размером 2 байта)
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    name_file = audiopath + str(num) + ".png"
    plt.savefig(name_file, bbox_inches="tight")
    plt.clf()
    return

def el_count (path):
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    return num_files

fds_0 = sorted(os.listdir('element'))
for img_0 in fds_0:
    name_d = 'element/' + img_0
    fds = sorted(os.listdir(name_d))
    for wav_file in fds:
        if wav_file.endswith(('.wav')):
            name_f = name_d + '/' + wav_file
            print(name_f)
            plotstft(1, name_f)



