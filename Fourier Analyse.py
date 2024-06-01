import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import tracemalloc
import time

# Funktion zur Fourier-Analyse in Blöcken und Spektrogramm-Anzeige
def block_fourier_analysis(file_path, block_size=256, shift=1, duration=60):
    tracemalloc.start()  # Speicherüberwachung starten

    # Lade die WAV-Datei
    sampling_rate, data = wavfile.read(file_path)

    # Falls die WAV-Datei stereo ist, wandle sie in mono um
    if len(data.shape) == 2:
        data = data[:, 0]

    # Begrenze die Daten auf die ersten 'duration' Sekunden
    n = min(len(data), int(sampling_rate * duration))
    data = data[:n]

    # Speicher für Ergebnisse
    time_blocks = []
    magnitude_blocks = []
    memory_usage = []

    # Gesamtanzahl der Iterationen berechnen
    total_iterations = (n - block_size + 1) // shift

    start_time = time.time()
    elapsed_time_intervals = []

    # Blöcke verarbeiten
    for i, start in enumerate(range(0, n - block_size + 1, shift)):
        block = data[start:start + block_size]
        fourier_transform = np.fft.fft(block)
        magnitude = np.abs(fourier_transform)

        time_blocks.append(start / sampling_rate)
        magnitude_blocks.append(magnitude[:block_size // 2])

        # Speicherverbrauch erfassen und ausgeben in jedem Loop
        current, _ = tracemalloc.get_traced_memory()
        elapsed_time = time.time() - start_time
        memory_usage.append((elapsed_time, current))
        print(f"Elapsed time: {elapsed_time:.2f}s, Current memory usage: {current / (1024 * 1024):.2f} MB")

        # Fortschrittsanzeige ausgeben
        #progress = (i + 1) / total_iterations * 100
        #print(f'Progress: {progress:.2f}%')

    tracemalloc.stop()  # Speicherüberwachung stoppen

    # Umwandlung der Magnitudenliste in ein 2D-Array für das Spektrogramm
    magnitude_array = np.array(magnitude_blocks).T

    return time_blocks, magnitude_array, memory_usage, sampling_rate, elapsed_time_intervals


# Plotten des Spektrogramms
def plot_spectrogram(time_blocks, magnitude_array, sampling_rate, block_size):
    plt.figure(figsize=(12, 8))
    extent = [time_blocks[0], time_blocks[-1], 0, sampling_rate / 2]
    plt.imshow(20 * np.log10(magnitude_array), aspect='auto', extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spektrogramm')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Frequenz (Hz)')
    plt.show()


# Plotten des Speicherverbrauchs
def plot_memory_usage(memory_usage):
    elapsed_time, current_usage, peak_usage = zip(*memory_usage)

    # Speicherverbrauch in Megabytes umrechnen
    current_usage_mb = np.array(current_usage) / (1024 * 1024)
    peak_usage_mb = np.array(peak_usage) / (1024 * 1024)

    plt.figure(figsize=(12, 6))
    plt.plot(elapsed_time, current_usage_mb, label='Aktueller Speicherverbrauch (MB)')
    #plt.plot(elapsed_time, peak_usage_mb, label='Maximaler Speicherverbrauch (MB)', linestyle='--')
    plt.title('Speicherverbrauch über die Laufzeit')
    plt.xlabel('Verstrichene Zeit (s)')
    plt.ylabel('Speicherverbrauch (MB)')
    plt.legend()
    plt.show()


# Datei und Parameter definieren
file_path = 'audio.wav'  # Ersetze dies durch den Pfad zu deiner heruntergeladenen Datei

# Beispiel für das Setzen von eigenen Werten für die Parameter
block_size = 256  # Blockgröße in Samples
shift = 1  # Anzahl der Samples, um die der Block bei jedem Schritt verschoben wird
dur = 60

# Fourier-Analyse durchführen (auf die ersten 60 Sekunden beschränkt)
# Hier werden die explizit gesetzten Werte verwendet
time_blocks, magnitude_array, memory_usage, sampling_rate, elapsed_time_intervals = block_fourier_analysis(file_path, block_size, shift, dur)

# Spektrogramm plotten
#plot_spectrogram(time_blocks, magnitude_array, sampling_rate, block_size)

# Speicherverbrauch plotten
#plot_memory_usage(memory_usage)
