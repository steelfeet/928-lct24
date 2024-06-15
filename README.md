# Команда "Мишки Гамми". ЛЦТ 2024
## Кейс №23 Нейросеть для мониторинга воздушного пространства вокруг аэропортов 

![изображение](https://github.com/steelfeet/928-lct24/assets/75137738/4d43f94a-1554-4c87-91c2-f5facb8ced0b)

Проект состоит из двух частей:
ver_1_yolo.py - детекция в режиме "Вахтер" (Минимизация ложно-положительных срабатываний детектора)
ver_2_gan.py - детекция в режиме "ПП" (Максимизация дальности обнаружения)

### Порядок установки ver_1_yolo
pip install -r requirements.txt
Обратите внимание, версия torch должна соответствовать вашей ОС и GPU. Поэтому сначала стоит установить нужную Вам версию torch, а затем пакеты из requirements.txt

### Порядок установки ver_2_gan
Надо добавить пакеты из [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN):
cd Real-ESRGAN
pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

### Запуск обеих версий
python ver_1_yolo.py -i images -o labels -conf 0.25  0.6
-i - имя zip-архива с изображениями (расширение указывать не надо), по умолчанию images
-o - имя получаемого zip-архива с метками (расширение указывать не надо), по умолчанию labels
-conf, -iou - gfhfvtnhs ltntrwbb 
