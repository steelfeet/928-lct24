# Команда "Мишки Гамми". ЛЦТ 2024
## Кейс №23 Нейросеть для мониторинга воздушного пространства вокруг аэропортов 

![изображение](https://github.com/steelfeet/928-lct24/assets/75137738/4d43f94a-1554-4c87-91c2-f5facb8ced0b)

Проект состоит из двух частей:  
ver_1_yolo.py - детекция в режиме "Вахтер" (Минимизация ложно-положительных срабатываний детектора)  
ver_2_gan.py - детекция в режиме "ПП" (Максимизация дальности обнаружения)  

### Порядок установки ver_1_yolo
[создаем и активируем виртуальное окружение](https://docs.python.org/3/library/venv.html)  
python -m venv /path/to/new/virtual/environment  
source myvenv/bin/activate  

Для Вашего удобства мы подготовили полностью работоспособное виртуальное окружение [gummy.zip](https://disk.yandex.ru/d/N8iylUdlmqCGtg). Его осталось только скачать и активировать командой  
source gummy/bin/activate  

**аккуратно инсталлируем требуемые модули**  
pip install -r requirements.txt  
Обратите внимание, **версия torch должна соответствовать вашей ОС и GPU**. Поэтому сначала стоит установить нужную Вам версию torch, а затем пакеты из requirements.txt

### Порядок установки ver_2_gan
Надо добавить пакеты из [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN):  
cd Real-ESRGAN  
pip install basicsr  
pip install facexlib  
pip install gfpgan  
pip install -r requirements.txt  
python setup.py develop  

Первый запуск обеих версий может потребовать выхода в Интернет для скачивания предобученных моделей.  
Затем небходимо [скачать новые файлы с весами для датасета had](https://disk.yandex.ru/d/N8iylUdlmqCGtg) и разместить их в том-же каталоге. 

При запуске ver_2_gan может возникать ошибка: "ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'". В этом случае [надо отредактировать файл](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985) Вашего виртуального окружения /lib/python3.12/site-packages/basicsr/data/degradations.py и заменить  
from torchvision.transforms.functional_tensor import rgb_to_grayscale  
на  
from torchvision.transforms.functional import rgb_to_grayscale  
В предложенном виртуальном окружении эта ошибка уже исправлена


### Запуск обеих версий
python ver_1_yolo.py -i images -o labels -conf 0.25 -iou 0.6  
-i - имя zip-архива с изображениями (расширение указывать не надо), по умолчанию images (в директории файла ver_1_yolo.py должен быть файл images.zip c набором изображений в корне архива)  
-o - имя получаемого zip-архива с метками (расширение указывать не надо), по умолчанию labels  
-conf, -iou - параметры детекции

При возникновении проблем, обращайтесь. 
