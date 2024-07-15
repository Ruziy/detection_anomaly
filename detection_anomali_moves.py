import cv2
import numpy as np

# Параметры для отслеживания
trackers = cv2.MultiTracker_create()

# Открытие видеопотока
cap = cv2.VideoCapture(r'C:\Users\Alex\Desktop\test_work_with_neyron\drafts_AI\work_with_book\open_code_TZ\real_TZ\video\test_video_anomaly.mkv')

# Чтение первого кадра
ret, frame = cap.read()
if not ret:
    print("Не удалось прочитать видео")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Изменение размера кадра до 1600x800 пикселей
frame = cv2.resize(frame, (1600, 800))

# Определение ROI
roi = cv2.selectROI("Выбор ROI", frame, fromCenter=False)
cv2.destroyWindow("Выбор ROI")

# Добавление отслеживателя
tracker = cv2.TrackerCSRT_create()
trackers.add(tracker, frame, roi)

# Параметры для аномального движения
initial_width = roi[2]  # Ширина ROI
initial_height = roi[3]  # Высота ROI
# threshold_speed = initial_width  # Порог скорости: ширина ROI
threshold_speed = 10
print(f"===============Speed thresh is : {threshold_speed}")
# height_threshold = initial_height / 3  # Порог высоты прыжка: треть высоты ROI
height_threshold=10
print(f"===============Jump thresh is : {height_threshold}")

# Начальная позиция
initial_positions = []
speeds = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Изменение размера кадра до 1600x800 пикселей
    frame = cv2.resize(frame, (1600, 800))

    # Обновление отслеживателей
    success, boxes = trackers.update(frame)
    
    for i, box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Расчет скорости и высоты прыжка
        center_x = x + w / 2
        center_y = y + h / 2

        speed = 0.0  # Инициализация переменной speed
        jump_height = 0.0  # Инициализация переменной jump_height

        if len(initial_positions) > i:
            prev_x, prev_y = initial_positions[i]
            speed = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
            speed = max(0.0, speed)
            speeds[i].append(speed)
            avg_speed = np.mean(speeds[i])

            if speed > threshold_speed:
                print(f"Now speed is : {speed}")
                cv2.putText(frame, "Anomaly speed!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            jump_height = center_y - prev_y
            jump_height = max(0.0, jump_height)
            if jump_height > height_threshold:
                print(f"Now jump is : {jump_height}")
                cv2.putText(frame, "Anomaly jump!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            initial_positions[i] = (center_x, center_y)
        else:
            initial_positions.append((center_x, center_y))
            speeds.append([])

        # Отображение текущей скорости и высоты прыжка
        cv2.putText(frame, f"Speed: {speed:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"Jump: {jump_height:.2f}", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Показ кадра
    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
