# AB-test-pipeline
Try to accumulate all steps for create full AB-test pipeline

TODO list
1. Для расчета размера выборки и эффекта нужно вычислить дисперсию целевой метрики
    - Чтобы снизить объем требуемых данных и снизить mde, нужно снизить дисперсию за счет использования стратификации входных данных и удаления выбросов;
    - При этом, данные по пользователям должны поступать вперемешку, без разбиения на тест и контроль;
    - Следовательно, структура датафрейма примет следующий вид:
        - id пользователя (int)
        - значение целевой метрики (int)
        - поле для стратификации 1 
        - поле для стратификации 2
        - поле для стратификации N
    !!! Необходимо сгенерировать такой df
    ----DONE
    
    TODO
    - Необходимо посчитать стратифицированные среднее и дисперсию по всей выборке, которая подается на вход функции. При этом, предполагается, что проведен пред анализ на генеральной совокупности и веса страт известны.
    
    Пример: отобрана выборка из 10к пользователей, для которой нужно посчитать стратифицированные оценки на основе заданых весов страт.

