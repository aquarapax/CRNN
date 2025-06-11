import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import random

# 1. Функции обработки изображений с OpenCV
def preprocess_image(image_path, img_height=32, min_width=100):
    """
    Улучшенная обработка изображения:
    - Улучшенное удаление шума
    - Оптимальная бинаризация
    - Эффективное удаление линий/сетки
    - Сохранение четкости текста
    """
    # Чтение изображения
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Перевод в серый с улучшенным контрастом
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Удаление шума (неразрушающее)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Улучшение контраста (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # Адаптивная бинаризация с оптимизированными параметрами
    binary = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
    
    # Удаление мелкого шума (морфологическое открытие)
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise, iterations=1)
    
    # Удаление крупных артефактов (линий/сетки)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    
    # Удаляем вертикальные линии
    vertical_removed = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_vertical)
    result = cv2.subtract(cleaned, vertical_removed)
    
    # Удаляем горизонтальные линии
    horizontal_removed = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_horizontal)
    final = cv2.subtract(result, horizontal_removed)
    
    # Ресайз с сохранением пропорций
    h, w = final.shape
    ratio = w / h
    new_w = int(img_height * ratio)
    resized = cv2.resize(final, (new_w, img_height), interpolation=cv2.INTER_CUBIC)
    
    # Дополнение до минимальной ширины
    if new_w < min_width:
        pad_width = min_width - new_w
        resized = np.pad(resized, ((0, 0), (0, pad_width)), 
                        mode='constant', constant_values=0)
    
    # Нормализация и добавление размерности канала
    normalized = (resized.astype(np.float32) / 255.0)
    return np.expand_dims(normalized, axis=0)

# 2. Создание датасета PyTorch
class CRNNDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        """
        Args:
            root_dir (string): Директория с изображениями.
            labels_file (string): Путь к TSV файлу с метками.
            transform (callable, optional): Дополнительные трансформации.
        """
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(labels_file, sep='\t', header=None, names=['file', 'label'])
        self.transform = transform
        
        # Приведение меток к нижнему регистру
        self.labels_df['label'] = self.labels_df['label'].str.lower()
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        
        # Обработка изображения
        image = preprocess_image(img_name)
        
        # Получение метки
        label = self.labels_df.iloc[idx, 1]
        
        # Конвертация в тензор
        image_tensor = torch.from_numpy(image)
        
        return image_tensor, label

# Функция для сохранения датасетов
def save_datasets(data_dir, train_labels='train.tsv', test_labels='test.tsv'):
    # Создание датасетов
    train_dataset = CRNNDataset(os.path.join(data_dir, 'train'), 
                               os.path.join(data_dir, train_labels))
    test_dataset = CRNNDataset(os.path.join(data_dir, 'test'), 
                              os.path.join(data_dir, test_labels))
    
    # Сохранение датасетов
    torch.save(train_dataset, os.path.join(data_dir, 'train_dataset.pt'))
    torch.save(test_dataset, os.path.join(data_dir, 'test_dataset.pt'))
    
    print(f"Датасеты сохранены в {data_dir}")
    return train_dataset, test_dataset

