# ============================================
# Author: Виктория Комарова (Vikitoria007)
# License: MIT
# GitHub: https://github.com/Vikitoria007/noisy-channel-transmission
# ============================================

import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 

# ==================================================
# СЛОВАРЬ СТАНДАРТОВ (параметры модуляции и диапазоны SNR)
# ==================================================
STANDARTS = {
    "MECH": {
        "name": "MECH (Mesh-сети)", 
        "modulation": "BPSK", 
        "snr_min": 0, 
        "snr_max": 12,
        "description": "Низкая скорость, большая дальность, высокая помехоустойчивость"
    }, 
    "MANET": {
        "name": "MANET (Мобильные ad-hoc сети)", 
        "modulation": "QPSK", 
        "snr_min": 0,
        "snr_max": 15,
        "description": "Средняя скорость, самоорганизация"
    },
    "FONBETA": {
        "name": "FONBETA (Wi-Fi offload)",
        "modulation": "16QAM", 
        "snr_min": 5, 
        "snr_max": 20, 
        "description": "Высокая скорость, зона покрытия" 
    }, 
    "ASTM_F38": {
        "name": "ASTM_F38 (Связь БПЛА)", 
        "modulation": "BPSK", 
        "snr_min": 0,
        "snr_max": 12, 
        "description": "Очень высокая помехоустойчивость, эхо-сигналы"
    }
}

# ==================================================
# ФУНКЦИИ МОДУЛЯЦИИ И ДЕМОДУЛЯЦИИ
# ==================================================
def modulate_bpsk(bits):
    """BPSK: бит 0 -> -1, бит 1 -> +1"""
    return np.array([1 if b == 1 else -1 for b in bits])

def demodulate_bpsk(symbols):
    """Демодуляция BPSK: символ > 0 -> 1, иначе 0"""
    return np.array([1 if s > 0 else 0 for s in symbols])

def modulate_qpsk(bits):
    """QPSK: группирует по 2 бита в комплексный символ"""
    symbols = [] 
    for i in range(0, len(bits) - 1, 2):
        b1, b2 = bits[i], bits[i + 1]

        if b1 == 0 and b2 == 0:
            symbols.append(1 + 1j)
        elif b1 == 0 and b2 == 1:
            symbols.append(-1 + 1j)
        elif b1 == 1 and b2 == 0:
            symbols.append(1 - 1j)
        else: 
            symbols.append(-1 - 1j) 
    return np.array(symbols)

def demodulate_qpsk(symbols): 
    """Демодуляция QPSK: обратное отображение в биты"""
    bits = []
    for s in symbols:
        if s.real > 0 and s.imag > 0:
            bits.extend([0, 0])
        elif s.real < 0 and s.imag > 0:
            bits.extend([0, 1])
        elif s.real > 0 and s.imag < 0:
            bits.extend([1, 0])
        else:
            bits.extend([1, 1])
    return np.array(bits) 

def modulate_16qam(bits):
    """16QAM: группирует по 4 бита в комплексный символ (сетка 4x4)"""
    symbols = [] 
    for i in range(0, len(bits) - 3, 4):
        b = bits[i:i+4] 
        val = b[0] * 8 + b[1] * 4 + b[2] * 2 + b[3] 
        re = (val % 4) - 1.5 
        im = (val // 4) - 1.5 
        symbols.append(re + 1j*im) 
    return np.array(symbols) 

def demodulate_16qam(symbols):
    """Демодуляция 16QAM: обратное отображение в биты"""
    bits = [] 
    for s in symbols:
        re = min(max(int((s.real + 1.5) // 1), 0), 3)
        im = min(max(int((s.imag + 1.5) // 1), 0), 3)
        val = im * 4 + re 
        bits.extend([(val >> 3) & 1, (val >> 2) & 1, (val >> 1) & 1, val & 1])
    return np.array(bits)

# ==================================================
# МОДЕЛЬ КАНАЛА
# ==================================================
def add_awgn_noise(signal, snr_db):
    """Белый гауссовский шум (AWGN)"""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) 
    return signal + noise 

def add_echo(signal, delay=5, attentuation=0.5):
    """Имитация многолучевого распространения (эхо) для ASTM F38"""
    echo = np.zeros(len(signal), dtype=complex)
    if len(signal) > delay:
        echo[delay:] = signal[:-delay] * attentuation 
        return signal + echo 
    return signal

# ==================================================
# РАСЧЁТ BER (Bit Error Rate)
# ==================================================
def calculate_ber(original_bits, received_bits):
    """Вычисление BER: количество ошибочных битов / общее число битов"""
    min_len = min(len(original_bits), len(received_bits))
    original_bits = original_bits[:min_len]
    received_bits = received_bits[:min_len]
    errors = np.sum(original_bits != received_bits)
    return errors / len(original_bits)

# ==================================================
# ПОСТРОЕНИЕ ГРАФИКА BER(SNR)
# ==================================================
def simulate_ber(standart_key, num_bits=10000):
    """Построение графика BER(SNR) для выбранного стандарта"""
    config = STANDARTS[standart_key]
    mod_type = config["modulation"]
    snr_min = config["snr_min"]
    snr_max = config["snr_max"]
    print(f"\n BER для {config['name']} ({mod_type})")

    bits = np.random.randint(0, 2, num_bits)

    if mod_type == "BPSK":
        symbols_tx = modulate_bpsk(bits)
        demod_func = demodulate_bpsk
    elif mod_type == "QPSK":
        bits = bits[:len(bits) - (len(bits) % 2)]
        symbols_tx = modulate_qpsk(bits)
        demod_func = demodulate_qpsk 
    else: 
        bits = bits[:len(bits) - (len(bits) % 4)]
        symbols_tx = modulate_16qam(bits)
        demod_func = demodulate_16qam
    
    snr_range = range(snr_min, snr_max)
    ber_values = []

    for snr in snr_range:
        symbols_rx = add_awgn_noise(symbols_tx, snr)

        if standart_key == "ASTM_F38":
            symbols_rx = add_echo(symbols_rx)

        received_bits = demod_func(symbols_rx)
        ber = calculate_ber(bits, received_bits)
        ber_values.append(ber)

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_range, ber_values, 'o-', linewidth=2)
    plt.xlabel('SNR dB')
    plt.ylabel('BER')
    plt.title(f'BER vs SNR - {config["name"]} ({mod_type})')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(1e-6, 1)
    plt.figtext(0.99, 0.01, "Vikitoria007 | github.com/Vikitoria007", ha='right', va='bottom', fontsize=7, alpha=0.6)
    plt.savefig(f'ber_{standart_key}.png', dpi=150)
    plt.show()

# ==================================================
# ВИЗУАЛИЗАЦИЯ СОЗВЕЗДИЯ
# ==================================================
def plot_constellation(standart_key, snr=10):
    """Рисует созвездие передатчика и приёмника"""
    config = STANDARTS[standart_key]
    mod_type = config["modulation"]

    if mod_type == "BPSK":
        bits = np.random.randint(0, 2, 1000)
        symbols_tx = modulate_bpsk(bits)
    elif mod_type == "QPSK":
        bits = np.random.randint(0, 2, 1000)
        bits = bits[:len(bits) - (len(bits) % 2)]
        symbols_tx = modulate_qpsk(bits)
    else:
        bits = np.random.randint(0, 2, 1000)
        bits = bits[:len(bits) - (len(bits) % 4)]
        symbols_tx = modulate_16qam(bits)

    symbols_rx = add_awgn_noise(symbols_tx, snr)
    if standart_key == "ASTM_F38":
        symbols_rx = add_echo(symbols_rx)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)  
    plt.scatter(symbols_tx.real, symbols_tx.imag, s=5, alpha=0.5)
    plt.title(f"{config['name']} - Передатчик")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(symbols_rx.real, symbols_rx.imag, s=5, alpha=0.5, c='red')
    plt.title(f"{config['name']} - Приемник (SNR={snr} dB)")
    plt.grid(True)
    plt.figtext(0.99, 0.01, "Vikitoria007 | github.com/Vikitoria007", ha='right', va='bottom', fontsize=7, alpha=0.6)
    plt.savefig(f'constellation_{standart_key}.png', dpi=150)
    plt.show()

# ==================================================
# ПРЕОБРАЗОВАНИЕ ИЗОБРАЖЕНИЙ В БИТЫ И ОБРАТНО
# ==================================================
def image_to_bits(image_path, size=(64, 64)):
    """Конвертирует изображение в черно-белый массив битов"""
    img = Image.open(image_path).convert('L')
    img.thumbnail(size)
    pixels = np.array(img) 

    bits = [] 
    for row in pixels:
        for pixel in row:
            bits.extend([int(b) for b in format(pixel, '08b')])
    return np.array(bits), pixels.shape 

def bits_to_image(bits, shape):
    """Восстанавливает изображение из массива битов"""
    expected = shape[0] * shape[1] 
    bits = bits[:expected * 8]

    pixels = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        val = int(''.join(str(b) for b in byte), 2)
        pixels.append(val) 

    img_array = np.array(pixels, dtype=np.uint8).reshape(shape)
    return Image.fromarray(img_array, mode='L')

# ==================================================
# ПЕРЕДАЧА ИЗОБРАЖЕНИЯ ЧЕРЕЗ КАНАЛ
# ==================================================
def transmit_image(standart_key, image_path, snr=15):
    """Передаёт изображение через выбранный стандарт, показывает результат"""
    print(f"\n Передача изображений через {STANDARTS[standart_key]['name']} (SNR = {snr} dB)")
    
    bits, shape = image_to_bits(image_path)
    config = STANDARTS[standart_key]
    mod_type = config["modulation"]

    if mod_type == "BPSK":
        symbols_tx = modulate_bpsk(bits)
        demod_func = demodulate_bpsk
    elif mod_type == "QPSK":
        bits = bits[:len(bits) - (len(bits) % 2)]
        symbols_tx = modulate_qpsk(bits)
        demod_func = demodulate_qpsk
    else:
        bits = bits[:len(bits) - (len(bits) % 4)]
        symbols_tx = modulate_16qam(bits)
        demod_func = demodulate_16qam

    symbols_rx = add_awgn_noise(symbols_tx, snr)
    if standart_key == "ASTM_F38":
        symbols_rx = add_echo(symbols_rx)

    received_bits = demod_func(symbols_rx)
    received_bits = received_bits[:len(bits)]
    recovered_img = bits_to_image(received_bits, shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(Image.open(image_path).convert('L'), cmap='gray')
    ax1.set_title("Исходное изображение")
    ax1.axis('off')

    ax2.imshow(recovered_img, cmap='gray')
    ax2.set_title(f"После канала({config['name']}, SNR={snr} dB)")
    ax2.axis('off')

    plt.tight_layout()
    plt.figtext(0.99, 0.01, "Vikitoria007 | github.com/Vikitoria007", ha='right', va='bottom', fontsize=7, alpha=0.6)
    plt.savefig(f'image_comparison_{standart_key}.png', dpi=150)
    plt.show()

# ==================================================
# ОСНОВНАЯ ПРОГРАММА (МЕНЮ)
# ================================================== 
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("👩‍💻 Автор: Виктория Комарова")
    print("🔗 GitHub: https://github.com/Vikitoria007")
    print("📜 Лицензия: MIT")
    print("=" * 60)

    print("=" * 60)
    print("Программная модель передачи данных")
    print("(MECH, MANET, FONBETA, ASTM F38)")
    print("=" * 60)

    print("Выберите действие:\n")
    print("1. Построить BER(SNR) для стандарта")
    print("2. Показать созвездие для стандарта")
    print("3. Передать изображение через стандарт")
    print("4. Запустить все для одного стандарта")
    print("5. Сравнить BER всех стандартов на одном графике")
    print("0. Выход")
    print("=" * 60)
    
    choice = input("\nВаш выбор (0-5): ").strip()
    
    if choice in ["1", "2", "3", "4"]:
        print("\n Выберите стандарт")
        print("1. MECH")
        print("2. MANET")
        print("3. FONBETA")
        print("4. ASTM F38")
        std_choice = input("Номер стандарта (1-4): ").strip()
        std_map = {"1": "MECH", "2": "MANET", "3": "FONBETA", "4": "ASTM_F38"}
        if std_choice not in std_map:
            print("Неверный выбор")
        else:
            standart = std_map[std_choice]
            if choice == "1":
                simulate_ber(standart)
            elif choice == "2":
                plot_constellation(standart, snr=10)
            elif choice == "3":
                img_path = input("Введите путь к изображению (например, test.png): ").strip()
                if not img_path:
                    print("Путь не указан")
                else:
                    transmit_image(standart, img_path, snr=15)
            elif choice == "4":
                simulate_ber(standart)
                plot_constellation(standart, snr=10)
                img_path = input("\n Введите путь к изображению (Enter - пропустить: ").strip()
                if img_path:
                    transmit_image(standart, img_path, snr=15)
    
    elif choice == "5":
        print("DEBUG: Начинаем построение общего графика...")
        plt.figure(figsize=(10, 6))
        for key in STANDARTS.keys():
            print(f"Обработка {key}...", flush=True)
            config = STANDARTS[key]
            mod_type = config["modulation"]
            snr_min, snr_max = config["snr_min"], config["snr_max"]
            bits = np.random.randint(0, 2, 1000)
            if mod_type == "BPSK":
                symbols_tx = modulate_bpsk(bits)
                demod_func = demodulate_bpsk
            elif mod_type == "QPSK":
                bits = bits[:len(bits) - (len(bits) % 2)]
                symbols_tx = modulate_qpsk(bits)
                demod_func = demodulate_qpsk
            else:
                bits = bits[:len(bits) - (len(bits) % 4)]
                symbols_tx = modulate_16qam(bits)
                demod_func = demodulate_16qam
            snr_range = range(snr_min, snr_max)
            ber_values = []
            for snr in snr_range:
                symbols_rx = add_awgn_noise(symbols_tx, snr)
                if key == "ASTM_F38":
                    symbols_rx = add_echo(symbols_rx)
                received_bits = demod_func(symbols_rx)
                ber = calculate_ber(bits, received_bits)
                ber_values.append(ber)
            plt.semilogy(snr_range, ber_values, 'o-', linewidth=2, label=config["name"])
        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        plt.title('Сравнение BER всех стандартов')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.figtext(0.99, 0.01, "Vikitoria007 | github.com/Vikitoria007", ha='right', va='bottom', fontsize=7, alpha=0.6)
        plt.savefig('ber_comparison_all.png', dpi=150)
        plt.show()
        input("Нажмите Enter для выхода...")
    
    elif choice == "0":
        print("До свидания!")
    else:
        print("Неверный выбор")
