from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from tqdm.notebook import tqdm
import seaborn as sns
import statsmodels as sm
import json
import itertools


from collections import namedtuple
import scipy.stats as sps
import statsmodels.stats.api as sms
from tqdm.notebook import tqdm as tqdm_notebook # tqdm – библиотека для визуализации прогресса в цикле
from collections import defaultdict
from statsmodels.stats.proportion import proportion_confint
import numpy as np
ExperimentComparisonResults = namedtuple('ExperimentComparisonResults',                                         
                                         ['pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound'])
import sys
from collections import defaultdict
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn import preprocessing
import hashlib
from typing import Tuple, List

np.random.seed(5)

plt.style.use('bmh')


class t2_ab_core:
    def __init__(self, design_yaml: dict) -> None:
        self.design_yaml = design_yaml
        
    def _read_design_dict(self, design_yaml: dict) -> dict: #TODO типизация yaml файла
        '''
        Считывает yaml файл с заданными параметрами дизайна теста.
        
        args:
            design_yaml - yaml файл
        return:
            dict - словарь параметров
        '''
        return dict


    def _get_target_metric_calc(self, metric_dict: dict) -> dict:
        '''
        Выделяет целевую метрику из заданных параметров.
        
        args:
            metric_dict - словарь с данными о метриках (собирается из design_dict)
        return:
            dict - словарь параметров
        '''
        return {key.split('_')[0]:val for key,val in metric_dict['target_metric_calc'].items()}
    
    
    def _get_help_metrics(self, metric_dict: dict) -> dict:
        '''
        Выделяет вспомогательные метрики из заданных параметров.
        
        args:
            metric_dict - словарь с данными о метриках (собирается из design_dict)
        return:
            dict - словарь параметров
        '''
        return {key.split('_')[0]:val for key,val in metric_dict['help_metric_calc'].items()} 
    
    
    def _get_grouped_columns(self, data: pd.DataFrame, metric_dict: dict, exception_list: list) -> Tuple[list, pd.DataFrame]:
        '''
        Собирает те поля датафрейма, по которым будет проводиться агрегация.
        
        args:
            data - датафрейм с данными для анализа,
            metric_dict - словарь с параметрами (где указаны метрики),
            exception_list - лист с полями для исключения (которые не будут участвовать в агрегации)
        return:
            column_for_grouped - лист с полями для группировки,
            data[df_cols] - датафрейм с выбранными полями
        '''
        try:
            simple_ratio_metrics = list(self._get_target_metric_calc(metric_dict).keys())
            help_metrics = list(self._get_help_metrics(metric_dict).keys())
            all_metrics = list(set(simple_ratio_metrics + help_metrics))
        except KeyError as error:
            raise KeyError(f'Проверьте корректность наименования поля: {error}')
        column_for_grouped = [i for i in [col for col in data.columns] if i not in all_metrics + exception_list]
        df_cols = column_for_grouped + list(set(simple_ratio_metrics + help_metrics))
        return column_for_grouped, data[df_cols]
    
    
    def print_json(self, x: dict) -> None:
        '''
        Выводит читабельный словарь.
        
        args:
            x - словарь
        return:
            None
        '''
        return print(json.dumps(x, indent=4))
    
    
    def _remove_outliers(self, data: pd.DataFrame, metric_name: str, right_quantile: float=0.99, left_quantile: float=0.01) -> pd.DataFrame:
        '''
        По значениям 1% и 99% персентилей выбранной метрики обрезает датафрейм.
        
        args:
            data - датафрейм с данными,
            metric_name - наименование поля метрики, по которой будут вычисляться персентили,
            right_quantile - правый хвост, по умолчанию - 99%,
            left_quantile - левый хвост, по умолчанию - 1%,
        return:
            data - обрезанный датафрейм
        '''
        
        left_bound = data[metric_name].quantile(left_quantile)
        right_bound = data[metric_name].quantile(right_quantile)
        return data[(data[metric_name] > left_bound) & (data[metric_name] < right_bound)]
    

    def delta_method(self, data: pd.DataFrame, column_for_grouped: list, metric_name: str, is_sample: bool=False) -> dict:
        '''
        Рассчитывает дисперсию Дельта-методом. Обычно применяется для ratio-метрик.
        
        args:
            data - датафрейм с данными,
            column_for_grouped - поля для группировки,
            metric_name - наименование поля с целевой метрикой,
            is_sample - если проводится выборочная оценка дисперсии - True, для дизайна - False
        return:
            info_dict - словарь с рассчитанными параметрами
        '''
        
        delta_df = data.groupby(column_for_grouped).agg({metric_name: ['sum', 'count']})
        n_users = len(delta_df)
        delta_df.columns = ['_'.join(col).strip() for col in delta_df.columns.values]
        array_x = delta_df[f'{metric_name}_sum'].values
        array_y = delta_df[f'{metric_name}_count'].values
        mean_x, mean_y = np.mean(array_x), np.mean(array_y)
        var_x, var_y = np.var(array_x), np.var(array_y)
        cov_xy = np.cov(array_x, array_y)[0, 1]
        var_metric = (
            var_x / mean_y ** 2
            - 2 * (mean_x / mean_y ** 3) * cov_xy
            + (mean_x ** 2 / mean_y ** 4) * var_y
        )
        if is_sample:
            var_metric = var_metric / n_users
            
        info_dict = {}
        info_dict['mean_x, mean_y'] = [mean_x, mean_y]
        info_dict['var_x, var_y'] = [var_x, var_y]
        info_dict['cov_xy'] = cov_xy
        info_dict['n_users'] = n_users
        info_dict['var_metric'] = var_metric
        info_dict['std_metric'] = np.sqrt(var_metric)
        
        return info_dict


    def _linearization_agg(self, data: pd.DataFrame, column_for_grouped: list, value_name: str) -> pd.DataFrame:
        '''
        Вспомогательная функция для агрегации данных для линеаризации.
        
        args:
            data - датафрейм,
            column_for_grouped - поля для агрегации,
            value_name - метрика
        return:
            df_lin - датафрейм
        '''

        df_count_metric = data.groupby(column_for_grouped).agg({value_name: 'count'}).rename(columns={value_name: f'{value_name}_lin_count'}).reset_index()
        df_sum_metric = data.groupby(column_for_grouped).agg({value_name: 'sum'}).rename(columns={value_name: f'{value_name}_lin_sum'}).reset_index()
        df_lin = pd.merge(df_count_metric, df_sum_metric, how='inner', on=column_for_grouped).fillna(0)
        return df_lin
    
    
    def _ratio_metric_calc(self, data: pd.DataFrame, key: str, val, column_for_grouped: list, postfix: str, kappa: float=None, is_history_data: bool=True, linearization_calc: bool=True) -> pd.DataFrame:
        '''
        Рассчитывает ratio-метрики в обычном и линеаризованном виде, если это необходимо.
        
        args:
            data - датафрейм,
            key - метрика,
            val - агрегирующие функции (может быть типа str или list),
            column_for_grouped - поля для агрегации,
            postfix - постфикс для наименования метрики,
            kappa - коэффициент kappa,
            is_history_data - True, если на вход подаются исторические данные, kappa будет рассчитана на истории,
            linearization_calc - True, если нужна линеаризация
        return:
            agg_df - датафрейм с рассчитанной метрикой
        '''
        # расчет ratio-метрики как она есть
        agg_df = data.groupby(column_for_grouped).agg({key: val}).reset_index()
        agg_df.columns = [''.join(col).strip() for col in agg_df.columns.values]
        numerator = f'{key}{val[0]}'
        denominator = f'{key}{val[1]}'
        agg_df[f'{key}_ratio_{postfix}'] = agg_df[numerator] / agg_df[denominator]
        ratio_columns = column_for_grouped + [f'{key}_ratio_{postfix}']
        
        if linearization_calc:
            # расчет линеаризованной метрики
            if kappa is None:
                if is_history_data:
                    df_kappa = self._linearization_agg(data, column_for_grouped, key)
                else:
                    control_df = data[data['group'] == 'control'] # наименование контрольной группы
                    df_kappa = self._linearization_agg(control_df, column_for_grouped, key)
                kappa = np.sum(df_kappa[f'{key}_lin_sum']) / np.sum(df_kappa[f'{key}_lin_count'])
                #print('kappa is: ', kappa)
            df_lin = self._linearization_agg(data, column_for_grouped, key)
            df_lin[f'{key}_linear_{postfix}'] = df_lin[f'{key}_lin_sum'] - kappa * df_lin[f'{key}_lin_count']
            linear_columns = column_for_grouped + [f'{key}_linear_{postfix}']
            agg_df = pd.merge(agg_df[ratio_columns], df_lin[linear_columns], how='left', on=column_for_grouped)
            all_columns = list(agg_df.columns)
            return agg_df[all_columns], kappa # ?
        else:
            return agg_df[ratio_columns]


    def _user_metric_calc(self, data: pd.DataFrame, column_for_grouped: list, key: str, val, postfix: str) -> pd.DataFrame:
        '''
        Рассчитывает пользовательские метрики.
        
        args:
            data - датафрейм,
            key - метрика,
            val - агрегирующие функции (может быть типа str или list),
            column_for_grouped - поля для агрегации,
            postfix - постфикс для наименования метрики
        return:
            user_metrics - датафрейм с рассчитанной метрикой
        '''
        
        user_metrics = data.groupby(column_for_grouped).agg({key: val}).rename(columns={key: f'{key}_{val}_{postfix}'}).reset_index()
        return user_metrics
    
     # решение для метрик - расчет пользовательских, ratio и линеаризованных в одном датафрейме
    def calc_metrics(self, data: pd.DataFrame, metric_dict: dict, column_for_grouped: list, kappa: float=None, is_history_data: bool=True) -> Tuple[pd.DataFrame, dict]:
        '''
        Запускает расчет метрик на основе заданных в дизайне формул.
        
        args:
            data - датафрейм,
            metric_dict - словарь,
            kappa - коэффициент каппа,
            column_for_grouped - поля для агрегации,
            is_history_data - True, если на вход подаются исторические данные
        return:
            agg_df - датафрейм с рассчитанными метриками
            kappa_dict - словарь со значениями рассчитанных коэффициентов каппа
        '''
        kappa_dict = {}
        for metric_type, metric in metric_dict.items():
            
            if metric_type == 'target_metric_calc':
                first_metric = True
                for key, val in metric.items():
                    if isinstance(val, list):
                        target_ratio_metrics, kappa = self._ratio_metric_calc(data, key, val, column_for_grouped, 'target', kappa, is_history_data)
                        if first_metric:
                            agg_df = target_ratio_metrics
                            first_metric = False
                        else:
                            agg_df = pd.merge(agg_df, target_ratio_metrics, how='left', on=column_for_grouped)
                        kappa_dict[key] = kappa
                        kappa = None
                    else:
                        user_metrics = self._user_metric_calc(data, column_for_grouped, key, val, 'target')
                        if first_metric:
                            agg_df = user_metrics
                            first_metric = False
                        agg_df = pd.merge(agg_df, user_metrics, how='left', on=column_for_grouped)
            if metric_type == 'help_metric_calc':
                for key, val in metric.items():
                    key = key.split('_')[0]
                    if isinstance(val, list):
                        help_ratio_metrics = self._ratio_metric_calc(data, key, val, column_for_grouped, kappa, is_history_data, 'help', linearization_calc=False)
                        agg_df = pd.merge(agg_df, help_ratio_metrics, how='left', on=column_for_grouped)
                    else:
                        help_user_metrics = self._user_metric_calc(data, column_for_grouped, key, val, 'help')
                        agg_df = pd.merge(agg_df, help_user_metrics, how='left', on=column_for_grouped)
        return agg_df, kappa_dict
    
    
    def data_distribution_plot(self, metrics: List[pd.Series], verbose: bool=True) -> None:
        '''
        Строит графики распределения для N метрик в одном рисунке.
        
        args:
            metrics - лист серий с метриками для отрисовки графиков,
            verbose - True, если нужно отобразить на рисунке параметры распределения
        return:
            None - рисунок
        '''
    
        fig, ax = plt.subplots()
        x = 0.02
        y = 0.65
        step_y = 0
        for hist in metrics:
            data_mean = round(np.mean(hist), 4)
            data_std = round(np.std(hist), 4)
            data_median = round(np.median(hist), 4)
            skewness_ = np.round(skew(hist), 4)
            kurtosis_ = np.round(kurtosis(hist), 4)
            lenght = len(hist)
            ax.hist(
                hist, 
                bins=300,
                alpha=0.5,
                density=True,
                label=[hist.name]
            )
            if verbose:
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)        

                textstr = '\n'.join((
                    f'{hist.name}',
                    r'$\mu=%.3f$' % (data_mean, ),
                    r'$\mathrm{median}=%.3f$' % (data_median, ),
                    r'$\sigma=%.3f$' % (data_std, ),
                    r'$length=%.0f$' % (lenght, ),
                    #r'$skewness=%.3f$' % (skewness_, ),
                    #r'$kurtosis=%.3f$' % (kurtosis_, ),
                    ))

                ax.text(x, y+step_y, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
            step_y += 0.3
            ax.axvline(x = data_mean, ymin = 0, ymax = 0.90, linestyle='--', color='red', linewidth=0.6)
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper right')
        plt.show()
        
    
    def get_sample_size_complex(
            self, mu: float=None, std: float=None, mde: float=None, n_groups: int=2, 
            target_share: float=0.5, r: float=1, alpha: float=0.05, beta: float=0.2
        ) -> Tuple[float, str]:
        '''
        Функция для расчета размера выборки при неравных группах.
        Возвращает sample_size для обычной пользовательской метрики, при заданных параметрах теста.
        ТАРГЕТНЫЕ ГРУППЫ ДОЛЖНЫ БЫТЬ РАВНЫ
        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            mde - мин. детектируемый эффект,
            n_groups - количество групп в тесте с учетом всех контрольных и таргетных,
            target_share - доля одной таргетной группы,
            r - отношение самой маленькой группы к самой большой группе,
            alpha - уровень ошибки I рода,
            beta - уровень ошибки II рода
        return: 
            sample_size - размер выборки для теста,
            'complex' - тип метода 
        '''
        
        t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
        t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)
        sample_ratio_correction = r + 2 + 1 / r
        comparisons = n_groups - 1
        mu_diff_squared = (mu - mu * mde)**2
        sample_size = (
            sample_ratio_correction * (
                (t_alpha + t_beta)**2) * (std**2)
        ) / (
            mu_diff_squared * (1 - target_share * (comparisons - 1))
        )
        if n_groups == 2:
            return int(np.ceil(sample_size / 2)), 'complex'
        else:
            return int(np.ceil(sample_size)), 'complex'
        
    
    def get_sample_size_standart(
            self, mu: float=None, std: float=None, mde: float=None, alpha: float=0.05, 
            beta: float=0.2
        ) -> Tuple[float, float]:
        '''
        Классическая формула расчета размера выборок для двух групп.
        
        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            mde - мин. детектируемый эффект,
            alpha - уровень ошибки I рода,
            beta - уровень ошибки II рода
        return:
            sample_size - размер выборки для теста,
            'standart' - тип метода
        '''
        t_alpha = abs(norm.ppf(1 - alpha / 2, loc=0, scale=1))
        t_beta = norm.ppf(1 - beta, loc=0, scale=1)
        
        mu_diff_squared = (mu - mu * mde) ** 2
        z_scores_sum_squared = (t_alpha + t_beta) ** 2
        disp_sum = 2 * (std ** 2)
        sample_size = int(
            np.ceil(
                z_scores_sum_squared * disp_sum / mu_diff_squared
            )
        )
        return sample_size, 'standart'
    
    
    def get_sample_size_matrix(
            self, sample_size_func: function=None, effect_bounds: np.array=None, 
            n_groups: int=2, target_share: float=0.5, r: float=1.0, plot: bool=True
        ) -> pd.DataFrame:
        '''
        Строит матрицу значений размера выборки в зависимости от mde.
        
        args:
            sample_size_func - функция расчета размера выборки,
            effect_bounds - границы mde,
            n_groups - количество групп,
            target_share - доля одной таргетной группы,
            r - отношение самой маленькой группы к самой большой группе,
            plot - True, если нужно построить график
        return:
            df_res - датафрейм
        '''
        effect_bounds = np.linspace(1.01, 1.2, num=20) # сделать отдельную функцию для ввода параметров
        res = []
        for eff in effect_bounds:
            sample_size, sample_size_method = sample_size_func(
                mu: float=None, std: float=None, mde: float=None, # откуда пойдут параметры?
                n_groups=n_groups, target_share=target_share, r=r
            )
            res.append((eff, sample_size, sample_size_method))
        df_res = pd.DataFrame(res, columns=['effects', 'sample_size', 'calc_method'])
        if plot:
            sns.set_style("whitegrid")
            sns.lineplot(df_res, x='sample_size', y='effects', hue='calc_method')
            return
        else:
            return df_res


    def get_MDE(
        self, mu: float=None, std: float=None, sample_size: int=None, n_groups: int=2, 
        target_share: float=0.5, r: float=1, alpha: float=0.05, beta: float=0.2
        )-> Tuple[float, float]:
        '''
        Возвращает MDE для обычной пользовательской метрики, при заданных параметрах теста. 
                                        ТАРГЕТНЫЕ ГРУППЫ ДОЛЖНЫ БЫТЬ РАВНЫ
        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            sample_size - размер выборки для теста (включает в себя все группы),
            n_groups - количество групп в тесте с учетом всех контрольных и таргетных,
            target_share - доля одной таргетной группы,
            r - отношение самой маленькой группы к самой большой группе,
            alpha - уровень ошибки I рода,
            beta - уровень ошибки II рода
        return: 
            mde, mde*100/mu - mde в абсолютных и относительных значениях
        '''
        
        t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
        comparisons = n_groups - 1
        t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)
        sample_ratio_correction = r+2+1/r
        mde = np.sqrt(sample_ratio_correction)*(t_alpha + t_beta) * std / np.sqrt(sample_size*(1-target_share*(comparisons-1)))
        return mde, mde*100/mu


    def get_effects_matrix(
        self, mde_func: function, sample_size_bounds: np.array=None, n_groups: int=2, 
        target_share: float=0.5, r: float=1.0, plot: bool=True) -> pd.DataFrame:
        '''
        Строит матрицу значений размера эффектов в зависимости от размера выборки.
        
        args:
            mde_func - функция расчета эффекта,
            sample_size_bounds - границы размера выборки,
            n_groups: количество групп в тесте с учетом всех контрольных и таргетных,
            target_share: доля одной таргетной группы,
            r: отношение самой маленькой группы к самой большой группе
        return:
            df_res - датафрейм
        '''
        res = []
        for size in sample_size_bounds:
            effect_abs, effect_percent = mde_func(
                mu=MEAN, std=STD, sample_size=size, n_groups=n_groups, # откуда пойдут параметры?
                target_share=target_share, r=r
            )
            res.append((size, effect_abs, effect_percent))
        df_res = pd.DataFrame(res, columns=['sample_size', 'effect_abs', 'effect_percent'])
        if plot:
            sns.set_style("whitegrid")
            sns.lineplot(df_res, x='sample_size', y='effects', hue='calc_method')
            return
        else:
            return df_res
        
    def get_confidence_interval(
        self, mu: float=None, std: float=None, sample_size: float=None, 
        confidence_level: float=0.95, tails: int=2
        ) -> Tuple[float, float]:
        '''
        Вычисляет доверительный интервал величины.

        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            sample_size - размер выборки для теста (включает в себя все группы),
            confidence_level - уровень доверительного интервала,
            tails - двусторонняя или односторонняя проверка
        return 
            (left_bound, right_bound): границы доверительного интервала.
        '''
        
        significance_level = 1 - confidence_level
        cum_probability = 1 - (significance_level / tails)
        z_star = norm.ppf(cum_probability)
        
        left_bound = mu - z_star * std / np.sqrt(sample_size)    
        right_bound = mu + z_star * std / np.sqrt(sample_size)        
        return left_bound, right_bound