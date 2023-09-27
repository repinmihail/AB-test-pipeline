# AB-test-pipeline
Try to accumulate all steps for create full AB-test pipeline

Notion:
Дисперсию ratio-метрики на историческом периоде оцениваем дельта-методом. Остается открытым вопрос в возможности использования дельта-метода и стратификации.

TODO local
1. Поля таблицы на вход предполагается именовать по типу user_id: int|date: str (YYYY-mm-dd)|feature1|feature2|feature3
2. В словарь параметров добавятся параметры после разработки Splitter'а с конфликтующими экспериментами
3. Доработать функцию стратификации так, чтобы деление было на не равные группы или вынести это в другую функцию
4. Сделать расчет матриц для неравных групп
5. Проверка того, что сформировались верно стратифицированные группы -- done
6. Считать дельту средних в абсолютном выражении между тестом и контролем
7. Стратификация cuped метрики на оценке теста (оптимизация функции)
8. Функция теста с возможностью поправки Холма на множественное тестирование
9. Работа с выбросами - ковариация для массивов разного размера

TODO global
1. Splitter - при отборе пользователей в тест необходимо учитывать конфликты экспериментов