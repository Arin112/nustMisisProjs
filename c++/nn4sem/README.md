## Домашняя работа по нейронным сетям, четвёртый семестр
По условию домашнего задания было нобходимо написать нейронную сеть и реализовать алгоритм обратного распространения ошибки для её обучения.

Реализовано данное домашнее задание на С++20 без использования дополнительных библиотек в виде нейронной сети классификатора.

Для тестирования был использован датасет [ирисы Фишера](https://ru.wikipedia.org/wiki/%D0%98%D1%80%D0%B8%D1%81%D1%8B_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0).

В результате тестирования accuracy составил ~95.3%.

Для компиляции исходного кода необходимо использовать компилятор с поддержкой С++20, команда компиляции может быть такой:
```g++ -std=c++2a -Ofast ./main.cpp```