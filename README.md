# AB-test-pipeline

Краткое описание принципа работы пакета в MVP версии:

1. Импортируем t2_ab_core и считываем тренировочные данные;
2. Заполняем поля словаря с дизайном теста;
3. Инициализируем экземпляр класса t2_ab_core, передав словарь с дизайном в качестве аргумента;
4. Для получения полного дизайна теста вызываем метод design_test у созданного экземпляра класса. Метод принимает на вход датафрейм с идентификаторами пользователей и метрикой(-ами), датафрейм с идентификаторами пользователей и стратами, а также список полей для исключения из датафрейма с метриками, которые важны для "ручного" анализа, но не важны для текущего дизайна (например, дата события);
5. Метод design_test возвращает датафрейм с отобранными пилотной и контрольной группами, а также выводит некоторый анализ. Также создается ряд полезных для анализа атрибутов: таблицы матриц эффектов и размеров групп (sample_size_matrix, effects_matrix), словарь для проверки качества стратификации групп (stratified_dict_check), словарь с параметрами распределений (params), датафрейм с рассчитанными метриками на истории (calc_metrics_df) и пр. Более подробное описание атрибутов будет... скоро :)
6. Непосредственно тест на текущем этапе проводится в полуавтоматическом режиме: для этого пользователь, используя методы экземпляра класса, самостоятельно собирает данные экспериментального периода, соединяет их с датафреймом отобранных на дизайне групп, рассчитывает метрики, строит распределение, при необходимости рассчитывает cuped метрику и проводит тест. Более подробно шаги показаны в демо файле.

Ограничения версии:
1. Возможность проводить только одномерные тесты - с одной целевой метрикой;
2. Отсутствие бутстрапа


Notion:
Дисперсию ratio-метрики на историческом периоде оцениваем Дельта-методом. 

TODO local
1. Поля таблицы на вход предполагается именовать по типу user_id: int|date: str (YYYY-mm-dd)|feature1|feature2|feature3;
2. В словарь параметров добавятся параметры после разработки Splitter'а с конфликтующими экспериментами;
3. Доработать функцию стратификации так, чтобы деление было на не равные группы или вынести это в другую функцию;
4. Сделать расчет матриц для неравных групп;
5. Считать дельту средних в абсолютном выражении между тестом и контролем;
6. Стратификация cuped метрики на оценке теста (оптимизация функции);
7. Функция теста с возможностью поправки Холма на множественное тестирование;
8. Работа с выбросами - ковариация для массивов разного размера;

TODO global
1. Splitter - при отборе пользователей в тест необходимо учитывать конфликты экспериментов


MIT License

Copyright (c) 2023 misha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
