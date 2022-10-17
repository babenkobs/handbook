PEP8 - стандарт написания кода на python

## Полезные функции
```
type(variable) - посмотреть тип данных
len(variable) - узнать длинну строки или массива
```

## Типы данных
В Python переменные неизменяемые. Поэтому там нет итераторов. Если переменной присваивается новое значение - создается новая переменная (ячейка памяти).

Массив - это изменяемая последовательность данных (мы можем изменять, добавлять и удалять данные в массиве). Массив использует квадратные скобки[].
Чтобы отобразить участок массива, можно использовать slice: array[3:10]
Кортеж - это неизменяемый массив. Создается с помощью фигурных скобок().
Множество - массив уникальных значений. Создается с помощью фигурных скобок.
Словарь - ассоциативный массив, где каждый ключ уникальный. Задается с помощью фигурных скобок.
```
english_to_russian = {
    "ball" : "мяч",
    "table" : "стол"
}
```

## Операции с массивами
```
myarray.append(‘some’) - добавляет элемент в массив
```
Операции со словарями
```
if ‘some key’ in mydict: - проверяет есть ли ключ в словаре

for english_word, russian_word in english_to_russian.items():
    print(f"Перевод слова {english_word} - {russian_word}")
```

## Операции со строками
```
Добавить текст с несколькими строками в переменную - “””some string…”””
Добавить форматированную строку message = f”Hi, {name}!!!”
message.startswith(“Hi”) - начинается ли строка со слова Hi
‘Bogdan’ in message - есть ли слово Bogdan в строке
message.isdigit() - проверяет все ли символы в строке являются числом
‘Some sentence’.lower().split() - преобразует всю строку в нижний регистр и разделит слова в массив.
```

## Операции с числами
```
5 % 2 - возвращает остаток от деления. В данном случае вернет 1.
```

```
If Else
if some:
	print(some)
elif some == 0 or some == -1:
	some = 1
else:
	some = -100
```

Для многострочного условия, условие нужно обернуть в скобки:
```
if (some1 == 1 and
	some2 == 2 and
	some3 == 3):
	print(‘equal’)
```

## Циклы
```
max = int()
some = [12, 23, ‘some’]
for index, item in enumerate(some):
	print(f”{index + 1}. Item = {item}”)
	if max < item:
		max = item
range(0, len(some), 2) - функция range создает секвенцию чисел с 0 до длинны массива (3) с шагом 2 (0.2). Удобно использовать в циклах, когда нужно изменить шаг перебора.
```

## Ввод данных в скрипт
```
message = input(“Type something: “) - В переменную message попадет то что напечатают в терминале или вызовут скрипт вот так - echo ‘Something’ | python script.py
import sys
date = sys.argv[1].split()
```
получение данных из аргументов командной строки

## Функции
```
def some_function(some_variable, variable_with_default=10):
	return variable_with_default * 2
```

Аргумент с дефолтным значением должен быть в конце.
Название функции должно начинаться с глагола (get_, do_, has_, can_, is_…)

## Класс
```
Class Circle:
	“””Circle cans calculate diametr, area…”””
	
	def __init__(self, radius: float):
		“””Initialization of circle”””
		self.radius = radius

myCircle = Circle()
```

Обычный метод имеет доступ к атрибутам экземпляра через переменную self.
Статический метод не имеет доступ к атрибутам экземпляра через self и к атрибутам класса через cls.
```
class User:
    # какие-то определения методов

    @staticmethod
    def is_full_name(name: str) -> bool:
        return len(name.split()) > 1
```

Метод класса имеет доступ к атрибутам класса, но не имеет доступа к атрибутам экземпляра класса
```
class User:
USERS_COUNT = 0

def __init__(self):
User.USERS_COUNT += 1

@classmethod
def how_much_users(cls):
return cls.USERS_COUNT
```

## Нейминг
Названия функций и переменных должны быть полными и понятными. Использовать английский язык.
```
def is_valid_url(url: string) -> bool:
	“””The function returns true if http response is 400 or 301”””
```
Указание типа принимаемого аргумента и типа возвращаемых данных являются необязательными, а информативными. Также комментарий в начале кода функции на саму работу функции не влияет, но все это удобно для работы в ide или редакторе кода.

## Модули и пакеты
Модуль это файл с расширением py.

Импортировать модуль и вызвать его функцию:
```
import tools

tools.greet(‘Vika’)
```

Импортировать функцию из модуля:
```
from tools import greet

greet(‘Olya’)
```

Импортировать функцию из модуля и назвать её иначе:
```
from tools import greet as greeting

greeting(‘Rita’)
```

Пакет это папка с модулями python. В пакете должен быть файл __init__.py.

Импортировать модуль из пакета:
```
from tools import greetings

greetings.say_hello(‘Marina’)
```

Импортировать функцию из пакета:
```
from tools.greetings import say_hello as hello

hello(‘Tamara’)
```

## TimeZone
Правильно считается использовать время (объект) с указанием временной зоны. В python для этого удобно использовать pytz:
```
import datetime
import pytz

timezone = pytz.timezone(“Europe/Kiev”)
now_tz = timezone.localize(datetime.datetime.now())
```

Для форматированного вывода даты/времени можно использовать strftime:
```
now_tz.strftime(“%d.%m.%Y”)
```

Для парсинга даты/времени из строки в объект datetime можно использовать strptime:
```
now = datetime.datetime.strptime(“22.03.2021”, “%d.%m.%Y”)
```

## Json
```
import json
from pprint import pprint

try:
json_object = json.loads(‘[{“some_key”: “some_value”}]’)
except ValueError as e:
	print(“Wrong json!”)
else:
	pprint(json_object)
```

## ENV Виртуальное окружение
Правильно сначала создать виртуальное окружение, а уже потом создавать проект.
```
python3 -m venv env - создает папку с виртуальным окружением python3
source env/bin/activate - запускает виртуальное окружение (выполняя bash скрипт)
source env/bin/activate.fish - запускает виртуальное окружение (выполняя fish скрипт)
```
## RE (regular expressions)
```
\d - любая цифра
[a-z] - любой символ в нижнем регистре
[^az] - любой символ кроме тех что в скобках (a и z)
[\S|\n\s] - любой символ, даже перевод строки
```

Есть ли в начале строки слеш:
```
import re

if re.search(‘^/’, ‘/some.html’):
    print(‘link’)
```

Регистронезависимое регулярное выражение:
```
re.search(‘^/’, ‘/some.html’, re.IGNORECASE)
```
