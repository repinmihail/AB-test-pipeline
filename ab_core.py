from scipy import stats
import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from tqdm.notebook import tqdm
import seaborn as sns
import statsmodels as sm
import json
import itertools
from datetime import datetime


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
from typing import Tuple, List, NamedTuple, Callable

np.random.seed(5)

plt.style.use('bmh')

titlesize = 15
labelsize = 15
legendsize = 12
xticksize = 15
yticksize = xticksize

plt.rcParams['legend.markerscale'] = 1.5     # the relative size of legend markers vs. original
plt.rcParams['legend.handletextpad'] = 0.5
plt.rcParams['legend.labelspacing'] = 0.4    # the vertical space between the legend entries in fraction of fontsize
plt.rcParams['legend.borderpad'] = 0.5       # border whitespace in fontsize units
plt.rcParams['font.size'] = 12
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['figure.figsize'] = (15, 6)

plt.rc('xtick', labelsize=xticksize)
plt.rc('ytick', labelsize=yticksize)
plt.rc('legend', fontsize=legendsize)


class t2_ab_core:
    
    def __init__(self, design_dict: dict=None) -> None:
        
        self.exp_name = design_dict['exp_name']
        self.exp_duration = design_dict['exp_duration']
        self.client = design_dict['client']
        self.groups_count = design_dict['groups_count']
        self.target_share = design_dict['target_share']
        self.r = design_dict['r']
        self.alpha = design_dict['alpha']
        self.beta = design_dict['beta']
        self.history_data_availability = design_dict['history_data_availability']
        self.data_source_type = design_dict['data_source_type']
        self.data_sources = design_dict['data_sources']
        self.sample_size = design_dict['sample_size']
        self.mde = design_dict['mde']
        self.metric_dict = design_dict['metric_dict']
        self.user_id_field_name = design_dict['user_id_field_name']
        self.stratification_cols = design_dict['stratification_cols']
        self.target_metric_count = [key for key,val in design_dict['metric_dict']['target_metric_calc'].items()]
        self.n_iter = design_dict['n_iter']
        self.weights = design_dict['weights']
        
        if design_dict['effect_bounds'] is None:
            self.effect_bounds = np.linspace(1.01, 1.2, num=20)
        else:
            self.effect_bounds = np.array(design_dict['effect_bounds'])
        if design_dict['sample_size_bounds'] is None:
            self.sample_size_bounds = np.arange(1000, 10000, 500)
        else:
            self.sample_size_bounds = np.array(design_dict['sample_size_bounds'])

    def get_target_metric_calc(self, metric_dict: dict) -> dict:
        '''
        Выделяет целевую метрику из заданных параметров.
        
        args:
            metric_dict - словарь с данными о метриках (собирается из design_dict)
        return:
            dict - словарь параметров
        '''
        return {key.split('_')[0]:val for key,val in metric_dict['target_metric_calc'].items()}
    
    
    def get_help_metrics(self, metric_dict: dict) -> dict:
        '''
        Выделяет вспомогательные метрики из заданных параметров.
        
        args:
            metric_dict - словарь с данными о метриках (собирается из design_dict)
        return:
            dict - словарь параметров
        '''
        return {key.split('_')[0]:val for key,val in metric_dict['help_metric_calc'].items()} 
    
    
    def get_grouped_columns(self, data: pd.DataFrame, metric_dict: dict, exception_list: list) -> Tuple[list, pd.DataFrame]:
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
            simple_ratio_metrics = list(self.get_target_metric_calc(metric_dict).keys())
            help_metrics = list(self.get_help_metrics(metric_dict).keys())
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
    
    
    def remove_outliers(self, data: pd.DataFrame, metric_name: str, right_quantile: float=0.99, left_quantile: float=0.01) -> pd.DataFrame:
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


    def linearization_agg(self, data: pd.DataFrame, column_for_grouped: list, value_name: str) -> pd.DataFrame:
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
    
    
    def ratio_metric_calc(self, data: pd.DataFrame, key: str, val, column_for_grouped: list, postfix: str, kappa: float=None, is_history_data: bool=True, linearization_calc: bool=True) -> pd.DataFrame:
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
        if (postfix == 'target') & linearization_calc==True:
            self.ratio_target_metric_as_is = f'{key}_ratio_{postfix}'
        
        if linearization_calc:
            # расчет линеаризованной метрики
            if kappa is None:
                if is_history_data:
                    df_kappa = self.linearization_agg(data, column_for_grouped, key)
                else:
                    control_df = data[data['group'] == 'control'] # наименование контрольной группы
                    df_kappa = self.linearization_agg(control_df, column_for_grouped, key)
                kappa = np.sum(df_kappa[f'{key}_lin_sum']) / np.sum(df_kappa[f'{key}_lin_count'])
                #print('kappa is: ', kappa)
            df_lin = self.linearization_agg(data, column_for_grouped, key)
            df_lin[f'{key}_linear_{postfix}'] = df_lin[f'{key}_lin_sum'] - kappa * df_lin[f'{key}_lin_count']
            linear_columns = column_for_grouped + [f'{key}_linear_{postfix}']
            agg_df = pd.merge(agg_df[ratio_columns], df_lin[linear_columns], how='left', on=column_for_grouped)
            all_columns = list(agg_df.columns)
            self.target_metric_name = f'{key}_linear_{postfix}'
            return agg_df[all_columns], kappa # ?
        else:
            return agg_df[ratio_columns]


    def user_metric_calc(self, data: pd.DataFrame, column_for_grouped: list, key: str, val, postfix: str) -> pd.DataFrame:
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
        if postfix == 'target':
            self.target_metric_name = f'{key}_{val}_{postfix}'
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
                        target_ratio_metrics, kappa = self.ratio_metric_calc(data, key, val, column_for_grouped, 'target', kappa=kappa, is_history_data=is_history_data)
                        if first_metric:
                            agg_df = target_ratio_metrics
                            first_metric = False
                        else:
                            agg_df = pd.merge(agg_df, target_ratio_metrics, how='left', on=column_for_grouped)
                        kappa_dict[key] = kappa
                        kappa = None
                    else:
                        user_metrics = self.user_metric_calc(data, column_for_grouped, key, val, 'target')
                        if first_metric:
                            agg_df = user_metrics
                            first_metric = False
                        else:
                            agg_df = pd.merge(agg_df, user_metrics, how='left', on=column_for_grouped)
            if metric_type == 'help_metric_calc':
                for key, val in metric.items():
                    key = key.split('_')[0]
                    if isinstance(val, list):
                        help_ratio_metrics = self.ratio_metric_calc(data, key, val, column_for_grouped, 'help', kappa=kappa, is_history_data=is_history_data, linearization_calc=False)
                        agg_df = pd.merge(agg_df, help_ratio_metrics, how='left', on=column_for_grouped)
                    else:
                        help_user_metrics = self.user_metric_calc(data, column_for_grouped, key, val, 'help')
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
        
    
    def get_sample_size_complex(self, mu: float, std: float, mde: float) -> Tuple[float, str]:
        '''
        Функция для расчета размера выборки при неравных группах.
        Возвращает sample_size для обычной пользовательской метрики, при заданных параметрах теста.
        ТАРГЕТНЫЕ ГРУППЫ ДОЛЖНЫ БЫТЬ РАВНЫ
        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            mde - мин. детектируемый эффект,
            groups_count - количество групп в тесте с учетом всех контрольных и таргетных,
            target_share - доля одной таргетной группы,
            r - отношение самой маленькой группы к самой большой группе,
            alpha - уровень ошибки I рода,
            beta - уровень ошибки II рода
        return: 
            sample_size - размер выборки для теста,
            'complex' - тип метода 
        '''
        
        t_alpha = stats.norm.ppf(1 - self.alpha / 2, loc=0, scale=1)
        t_beta = stats.norm.ppf(1 - self.beta, loc=0, scale=1)
        sample_ratio_correction = self.r + 2 + 1 / self.r
        comparisons = self.groups_count - 1
        mu_diff_squared = (mu - mu * mde)**2
        sample_size = (
            sample_ratio_correction * (
                (t_alpha + t_beta)**2) * (std**2)
        ) / (
            mu_diff_squared * (1 - self.target_share * (comparisons - 1))
        )
        if self.groups_count == 2:
            return int(np.ceil(sample_size / 2)), 'complex'
        else:
            return int(np.ceil(sample_size)), 'complex'
        
    
    def get_sample_size_standart(self, mu: float, std: float, mde: float) -> Tuple[float, str]:
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
        t_alpha = abs(norm.ppf(1 - self.alpha / 2, loc=0, scale=1))
        t_beta = norm.ppf(1 - self.beta, loc=0, scale=1)
        
        mu_diff_squared = (mu - mu * mde) ** 2
        z_scores_sum_squared = (t_alpha + t_beta) ** 2
        disp_sum = 2 * (std ** 2)
        sample_size = int(
            np.ceil(
                z_scores_sum_squared * disp_sum / mu_diff_squared
            )
        )
        return sample_size, 'standart'
    
    
    def get_sample_size_matrix(self, sample_size_func: Callable, mu: float, std: float, plot: bool=False) -> pd.DataFrame:
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
        res = []
        for eff in self.effect_bounds:
            sample_size, sample_size_method = sample_size_func(mu, std, eff)
            res.append((eff, sample_size, sample_size_method))
        df_res = pd.DataFrame(res, columns=['effects', 'sample_size', 'calc_method'])
        if plot:
            sns.set_style("whitegrid")
            sns.lineplot(df_res, x='sample_size', y='effects', hue='calc_method')
            return
        else:
            return df_res


    def get_MDE(self, mu: float, std: float, sample_size: int) -> Tuple[float, float]:
        '''
        Возвращает MDE для обычной пользовательской метрики, при заданных параметрах теста. 
                                        ТАРГЕТНЫЕ ГРУППЫ ДОЛЖНЫ БЫТЬ РАВНЫ
        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            sample_size - размер выборки для теста (включает в себя все группы),
            groups_count - количество групп в тесте с учетом всех контрольных и таргетных,
            target_share - доля одной таргетной группы,
            r - отношение самой маленькой группы к самой большой группе,
            alpha - уровень ошибки I рода,
            beta - уровень ошибки II рода
        return: 
            mde, mde*100/mu - mde в абсолютных и относительных значениях
        '''
        
        t_alpha = stats.norm.ppf(1 - self.alpha / 2, loc=0, scale=1)
        comparisons = self.groups_count - 1
        t_beta = stats.norm.ppf(1 - self.beta, loc=0, scale=1)
        sample_ratio_correction = self.r + 2 + 1 / self.r
        mde = np.sqrt(sample_ratio_correction) * (t_alpha + t_beta) * std / np.sqrt(sample_size*(1-self.target_share*(comparisons-1)))
        return mde, mde*100/mu


    def get_effects_matrix(self, mde_func: Callable, mu: float, std: float, plot: bool=False) -> pd.DataFrame:
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
        for size in self.sample_size_bounds:
            effect_abs, effect_percent = mde_func(mu, std, size)
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


    def stratified_without_group(
        self, data: pd.DataFrame, strata_list: list, group_size: int, 
        weight: dict, metric_name: str, strat_concat_column: str='strat_concat_column', 
        group_count: int=1, verbose: bool=False
        ) -> Tuple[pd.DataFrame, float, float]:
        '''
        Случайным образом выбирает пользователей в зависимости от веса страты. 
        Рассчитывает стратифицированные оценки среднего и дисперсии.
        
        args:
            data - датафрейм,
            strata_list - страта в формате List[str(strata)],
            group_size - размер выборки,
            weight - веса страт, если известны заранее,
            metric_name - наименование целевой метрики,
            strat_concat_column - поле страт,
            group_count - количество групп,
            verbose - True, если нужен детальный вывод случайных величин
        return:
            random_strata_df - датафрейм страты,
            strat_mean - среднее страты,
            strat_var - дисперсия страты
        '''
        strat_df = data[data[strat_concat_column].isin(strata_list)].reset_index(drop=True)
        one_group_size = int(round(group_size * weight)) * group_count
        try:
            random_indexes_for_one_group = np.random.choice(np.arange(len(strat_df)), one_group_size, False)
        except ValueError:
            print('Объем выборки превышает объем входных данных.')
        random_strata_df = strat_df.iloc[random_indexes_for_one_group,:]
        strat_mean = random_strata_df.groupby('strat_concat_column')[metric_name].mean() * weight
        strat_var = random_strata_df.groupby('strat_concat_column')[metric_name].var() * weight
        if verbose:
            print(f'Размер датафрейма страты {strata_list}: {len(strat_df)}')
            print(f'one_group_size страты {strata_list}: {one_group_size}')
            print(f'Количество случайных индексов, всего: {len(random_indexes_for_one_group)}')
            print(f'Количество отобранных индексов: {len(random_indexes_for_one_group)}')
            print(f'Количество строк в датафрейме страты: {len(random_strata_df)}')
        return random_strata_df, strat_mean, strat_var


    def get_stratified_data(
        self, data: pd.DataFrame=None, group_size: int=None, metric_name: str=None,
        larger_group_weight: float=None, weights: float=None, seed: int=None, verbose: bool=False
        ) -> Tuple[pd.DataFrame, float, float]:
        '''
        Выравнивает общий датафрейм в соответствии с весами, либо переданными в качестве аргумента, либо рассчитанными по датафрейму.

        args:
            data - датафрейм с описанием объектов, содержит атрибуты для стратификации,
            group_size - размер одной группы,
            group_count - количество групп (по умолчанию 2),
            weights - словарь весов страт {strat: weight}
                    , где strat - страта (strat_concut_columns: str), weight - вес (float)
                    Если None, определить веса пропорционально доле страт в датафрейме data.
            seed - int, исходное состояние генератора случайных чисел для воспроизводимости
                результатов. Если None, то состояние генератора не устанавливается.
        return:
            stratified_data - один датафрейм того же формата, что и data, количество строк = group_size * group_count
        '''
        np.random.seed(seed)
        
        stratified_df = pd.DataFrame(columns=data.columns)
        stratified_mean = []
        stratified_var = []
        if weights:
            for strata, weight in weights.items():
                strata_list = [strata]
                try:
                    one_strata_df, one_strata_mean, one_strata_var = self.stratified_without_group(data, strata_list, group_size, weight, metric_name)
                    stratified_df = pd.concat([stratified_df, one_strata_df], ignore_index=True)
                    stratified_mean.append(one_strata_mean)
                    stratified_var.append(one_strata_var)
                except UnboundLocalError:
                    print('Нарушена индексация. Проверьте границы входимости индексов в датафрейм')
                    return
        else:
            len_data = len(data)
            strat_dict = data.groupby('strat_concat_column').count().iloc[:,0].to_dict()
            strat_dict_shares = {strata:share/len_data for (strata, share) in strat_dict.items()}
            for strata, weight in strat_dict_shares.items():
                strata_list = [strata]
                try:
                    one_strata_df, one_strata_mean, one_strata_var = self.stratified_without_group(data, strata_list, group_size, weight, metric_name)
                    stratified_df = pd.concat([stratified_df, one_strata_df], ignore_index=True)
                    stratified_mean.append(one_strata_mean)
                    stratified_var.append(one_strata_var)
                except UnboundLocalError:
                    print('Нарушена индексация. Проверьте границы входимости индексов в датафрейм')
                    return  
        return stratified_df, np.sum(stratified_mean), np.sum(stratified_var)


    def get_random_group(self, data: pd.DataFrame=None, group_size: int=None, seed: int=None) -> pd.DataFrame:
        '''
        Формирует случайные группы пользователей.
        
        args:
            data - датафрейм,
            group_size - размер группы,
            seed - int, если нужно зафикисировать случайный порядок
        return:
            data - датафрейм
        '''
        np.random.seed(seed)
        random_indexes_for_one_group = np.random.choice(np.arange(len(data)), group_size, False)
        return data.iloc[random_indexes_for_one_group,:]


    def get_stratified_with_groups(
        self, data: pd.DataFrame=None, group_size: int=None, weights: dict=None, seed: int=None
        ) -> pd.DataFrame:
        '''
        В соответствии с весами (заданными или рассчитанными) случайным образом формирует контрольную и пилотную группы.
        
        args:
            data - исходный датафрейм,
            group_size - размер группы,
            weights - вес группы,
            seed - int, если нужно зафикисировать случайный порядок
        return:
            data - датафрейм из пилотной и контрольной групп
        '''
        pilot = pd.DataFrame(columns=data.columns)
        control = pd.DataFrame(columns=data.columns)
        for strat, weight in weights.items():
            strat_df = data[data['strat_concat_column'].isin([strat])].reset_index(drop=True)
            ab_group_size = int(round(group_size * weight))
            random_indexes_ab = np.random.choice(np.arange(len(strat_df)), ab_group_size * 2, False)
            a_indexes = random_indexes_ab[:ab_group_size]
            b_indexes = random_indexes_ab[ab_group_size:]
            a_random_strata_df = strat_df.iloc[a_indexes,:]
            b_random_strata_df = strat_df.iloc[b_indexes,:]
    
            control = pd.concat([control, a_random_strata_df], ignore_index=True)
            pilot = pd.concat([pilot, b_random_strata_df], ignore_index=True)
        
        control['group'] = 'control'
        pilot['group'] = 'pilot'
        
        self.control_strat_dict = control.groupby('strat_concat_column').count().iloc[:,0].to_dict()
        self.pilot_strat_dict = pilot.groupby('strat_concat_column').count().iloc[:,0].to_dict()
        
        return pd.concat([control, pilot]).reset_index(drop=True)


    def get_stratified_groups(self, data: pd.DataFrame, group_size: int, weights: dict=None, seed: int=None) -> pd.DataFrame:
        '''
        Подбирает стратифицированные группы для эксперимента из заранее подготовленной стратифицированной выборки

        args:
            data - датафрейм с описанием объектов,
            group_size - int, размер одной группы,
            weights - dict, словарь весов страт {strat: weight}, где strat - str/int, weight - float.
                    Если None, то веса определяются пропорционально доле страт в датафрейме data.
            seed - int, если нужно зафикисировать случайный порядок
        return:
            df - датафрейм того же формата, что и data c пилотной и контрольной группами.
        '''
        np.random.seed(seed)
        self.stratified_dict_check = {}
        if weights:
            df = self.get_stratified_with_groups(data, group_size, weights=None, seed=None)
            self.stratified_dict_check['user_weights'] = weights
            self.stratified_dict_check['control_group_weights'] = self.control_strat_dict
            self.stratified_dict_check['pilot_group_weights'] = self.pilot_strat_dict
            return df
        else:
            strat_dict = data.groupby('strat_concat_column').count().iloc[:,0].to_dict()
            len_data = len(data)
            strat_dict_shares = {strata:share/len_data for (strata, share) in strat_dict.items()}
            df = self.get_stratified_with_groups(data, group_size, weights=strat_dict_shares, seed=None)
            self.stratified_dict_check['calc_weights'] = strat_dict
            self.stratified_dict_check['control_group_weights'] = self.control_strat_dict
            self.stratified_dict_check['pilot_group_weights'] = self.pilot_strat_dict
            return df
        

    def absolute_ttest(self, control: pd.Series=None, test: pd.Series=None) -> NamedTuple:
        '''
        Функция абсолютного t-теста. Выяисляет значения pvalue, границы доверительного интервала, его длину и эффект.
        
        args:
            control - контрольная группа,
            test - пилотная группа
        return:
            pvalue - значение pvalue, 
            effect - разница между средними значениями в тестовой и контрольной группах,
            ci_length - длина доверительного интервала,
            left_bound - левая граница доверительного интервала,
            right_bound - правая граница доверительного интервала
        '''
        
        mean_control = np.mean(control)
        mean_test = np.mean(test)
        var_mean_control  = np.var(control) / len(control)
        var_mean_test  = np.var(test) / len(test)
        
        difference_mean = mean_test - mean_control
        difference_mean_var = var_mean_control + var_mean_test
        difference_distribution = sps.norm(loc=difference_mean, scale=np.sqrt(difference_mean_var))

        left_bound, right_bound = difference_distribution.ppf([0.025, 0.975])
        ci_length = (right_bound - left_bound)
        pvalue = 2 * min(difference_distribution.cdf(0), difference_distribution.sf(0))
        effect = difference_mean
        return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound)

    
    def simple_ttest(self, control: pd.Series=None, test: pd.Series=None) -> float:
        '''
        Простая функция расчета t-теста из модуля stats.
        
        args:
            control - контрольная группа,
            test - пилотная группа
        return:
            pvalue - значение pvalue
        '''
        return stats.ttest_ind(control, test).pvalue


    def aa_ab_test_history_calc(
        self, data: pd.DataFrame=None, sample_size: int=None, target_metric: str=None, 
        effect_size_abs: float=None, test_type: Callable=None
        ) -> Tuple[np.array, np.array]:
        '''
        Функция для проведения ААB теста на истрических данных для проверки корректности подобранного статистического критерия.
        
        args:
            data - датафрейм,
            sample_size - размер групп,
            target_metric - наименование целевой метрики,
            effect_size_abs - размер эффекта в абсолютной величине,
            test_type - тип теста,
            n_iter - количество тестов (по умолчанию 10_000)
        return:
            pvalues_aa - массив результатов оценки АА теста,
            pvalues_ab - массив результатов оценки АВ теста
        '''
        pvalues_aa = []
        pvalues_ab = []
        
        for i in tqdm_notebook(range(self.n_iter)):
            random_inds = np.random.choice(len(data), sample_size * 2, False)
            random_a = random_inds[:sample_size]
            random_b = random_inds[sample_size:]
            
            control = data[target_metric][random_a]
            test_aa = data[target_metric][random_b]
            
            if effect_size_abs:
                test_ab = test_aa + effect_size_abs
            else:
                test_ab = test_aa + 0
            pvalue_aa = test_type(control, test_aa)
            pvalue_ab = test_type(control, test_ab)
            
            pvalues_aa.append(pvalue_aa)
            pvalues_ab.append(pvalue_ab)
        
        return np.array(pvalues_aa), np.array(pvalues_ab)


    def estimate_ci_bernoulli(self, p: float=None, n: float=None, alpha=0.05) -> Tuple[float, float]:
        '''
        Доверительный интервал для Бернуллиевской случайной величины.
        
        args:
            p - оценка вероятности ошибки (I или II рода),
            n - длина массива,
            alpha - уровень ошибки I рода
        return:
            Tuple[float, float] - границы доверительного интервала
        '''
        
        t = stats.norm.ppf(1 - self.alpha / 2, loc=0, scale=1)
        std_n = np.sqrt(p * (1 - p) / n)
        return p - t * std_n, p + t * std_n
    
    
    def print_estimated_errors(self, pvalues_aa: np.array, pvalues_ab: np.array) -> None:
        '''
        Оценивает вероятности ошибок.
        
        args:
            pvalues_aa - массив результатов оценки АА теста,
            pvalues_ab - массив результатов оценки АВ теста,
            alpha - уровень ошибки I рода
        return:
            None
        '''
        
        estimated_first_type_error = np.mean(pvalues_aa < self.alpha)
        estimated_second_type_error = np.mean(pvalues_ab >= self.alpha)
        ci_first = self.estimate_ci_bernoulli(estimated_first_type_error, len(pvalues_aa))
        ci_second = self.estimate_ci_bernoulli(estimated_second_type_error, len(pvalues_ab))
        print(f'Оценка вероятности ошибки I рода = {estimated_first_type_error:0.4f}')
        print(f'    Доверительный интервал = [{ci_first[0]:0.4f}, {ci_first[1]:0.4f}]')
        print(f'Оценка вероятности ошибки II рода = {estimated_second_type_error:0.4f}')
        print(f'    Доверительный интервал = [{ci_second[0]:0.4f}, {ci_second[1]:0.4f}]')
        
    
    def plot_pvalue_distribution(self, pvalues_aa: np.array, pvalues_ab: np.array) -> None:
        '''
        Рисует графики распределения p-value.
        
        args:
            pvalues_aa - массив результатов оценки АА теста,
            pvalues_ab - массив результатов оценки АВ теста,
            alpha - уровень ошибки I рода,
            beta - уровень ошибки II рода
        return:
            None
        '''
        estimated_first_type_error = np.mean(pvalues_aa < self.alpha)
        estimated_second_type_error = np.mean(pvalues_ab >= self.alpha)
        y_one = estimated_first_type_error
        y_two = 1 - estimated_second_type_error
        X = np.linspace(0, 1, 1000)
        Y_aa = [np.mean(pvalues_aa < x) for x in X]
        Y_ab = [np.mean(pvalues_ab < x) for x in X]

        plt.plot(X, Y_aa, label='A/A')
        plt.plot(X, Y_ab, label='A/B')
        plt.plot([self.alpha, self.alpha], [0, 1], '--k', alpha=0.8)
        plt.plot([0, self.alpha], [y_one, y_one], '--k', alpha=0.8)
        plt.plot([0, self.alpha], [y_two, y_two], '--k', alpha=0.8)
        plt.plot([0, 1], [0, 1], '--k', alpha=0.8)

        plt.title('Оценка распределения p-value', size=16)
        plt.xlabel('p-value', size=12)
        plt.legend(fontsize=12)
        plt.show()
        
    def calculate_theta(self, y_prepilot: np.array=None, y_pilot: np.array=None) -> float:
        '''
        Вычисляем Theta.

        args:
            y_pilot - значения метрики во время пилота,
            y_prepilot - значения ковариант (той же самой метрики) на препилоте
        return:
            theta - значение коэффициента тета
        '''
        covariance = np.cov(y_prepilot.astype(float), y_pilot.astype(float))[0, 1]
        variance = np.var(y_prepilot)
        theta = covariance / variance
        return theta

    def calculate_metric_cuped(
        self, df_history: pd.DataFrame=None, df_experiment: pd.DataFrame=None, user_id_name: str=None, 
        metric_name: str=None, theta: float=None) -> pd.DataFrame:
        '''
        Вычисляет коварианту и преобразованную метрику cuped.

        args:
            df - pd.DataFrame, датафрейм с данными по пользователям, метрикам (нормализованной ключевой метрикой) и стратами с разметкой: 
                1) на контроль и пилот (A/B/C..., где A-контроль) - столбец group,
                2) пред-экспериментальный и экспериментальный периоды (pilot/prepilot) - столбец period,
            user_id_name - str, название столбца с идентификаторами пользователей,
            metric_name - str, название полученной метрики
        return:
            res - датафрейм
        '''
        prepilot_period = df_history[df_history['period'] == 'history'].sort_values(user_id_name)
        pilot_period = df_experiment[df_experiment['period'] == 'pilot'].sort_values(user_id_name)
        
        if theta is None:
            theta = self.calculate_theta(
                np.array(prepilot_period[metric_name]), 
                np.array(pilot_period[metric_name])
                )
        res = pd.merge(
            prepilot_period,
            pilot_period[[user_id_name, metric_name]],
            how='inner', 
            on=user_id_name
        )
        cols = list(prepilot_period.columns)
        print('Theta is: ', theta)
        res.columns = cols + [f'{metric_name}_prepilot']
        res[f'{metric_name}_cuped'] = res[metric_name] - theta * res[f'{metric_name}_prepilot']
        
        return res
    
    
    def starta_merged(self, data: pd.DataFrame, strats: pd.DataFrame, user_id: str=None) -> pd.DataFrame:
        '''
        Соединяет метрику и страты, создает поле strat_concat_column.
        
        args:
            data - датафрейм с метрикой,
            strats - датафрейм со стратами,
            user_id - поле с идентификаторами пользователей, ключ
        return:
            res - датафрейм со стратами
        '''
        
        res = pd.merge(data, strats, on=user_id, how='inner')
        res['strat_concat_column'] = res[self.stratification_cols].astype('str').agg('-'.join, axis=1)
        res['period'] = 'history'
        return res
    
    
    def remove_outs(self, data: pd.DataFrame, exception_list: list) -> Tuple[list, pd.DataFrame]:
        '''
        Обработка выбросов, если целевая метрика одна, и формирует поля для группировки.
        
        args:
            data - датафрейм с метрикой,
            exception_list - лист для исключения полей (например, дата) содержит поля, которые не будут учтены в агрегации
        return:
            column_for_grouped - поля для агрегации,
            remove_outliers_df - датафрейм без выбросов
        '''
        self.column_for_grouped, df_ness_cols = self.get_grouped_columns(data, self.metric_dict, exception_list)
        if len(self.target_metric_count) == 1:
            remove_outliers_df = self.remove_outliers(df_ness_cols, self.target_metric_count[0])
        else:
            remove_outliers_df = df_ness_cols
        return self.column_for_grouped, remove_outliers_df
    
    
    def calc_distribution_params(self, remove_outliers_df: pd.DataFrame, strata_df: pd.DataFrame, calc_metrics_df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
        '''
        Рассчитывает параметры распределения целевых(-ой) метрик(-и).
        
        args:
            remove_outliers_df - датафрейм с метрикой,
            strata_df - датафрейм со стратами,
            exception_list - лист для исключения полей (например, дата) содержит поля, которые не будут учтены в агрегации
        return:
            params_dict - словарь с параметрами,
            remove_outliers_df - датафрейм без выбросов
        '''
        params_dict = {}
        # делаем предположение, что целевая метрика будет одна, потом реализуем для нескольких целевых метрик
        for metric_name, formula in self.metric_dict['target_metric_calc'].items():
            one_metric_params = {}
            try:
                if isinstance(int(metric_name.split('_')[-1]), int):
                    corr_metric_name = metric_name[:-2]
            except ValueError:
                corr_metric_name = metric_name
                pass
            if isinstance(formula, list):
                ratio_metrics_dict = self.delta_method(remove_outliers_df, self.column_for_grouped, corr_metric_name)
                MEAN = np.mean(abs(self.calc_metrics_df[self.ratio_target_metric_as_is]))
                VAR = ratio_metrics_dict['var_metric']
                STD = ratio_metrics_dict['std_metric']
                metric_type = 'ratio'
            if isinstance(formula, str):
                print('    Подтягиваю страты...')
                stratification_df = self.starta_merged(calc_metrics_df, strata_df)
                n_users = len(stratification_df[self.user_id_field_name].unique()) - 1
                print('Рассчитываю стратифицированные оценки...')
                _, MEAN, VAR = self.get_stratified_data(stratification_df, n_users, self.target_metric_name)
                STD = np.sqrt(VAR)
            one_metric_params['mean'] = MEAN
            one_metric_params['var'] = VAR
            one_metric_params['std'] = STD
            params_dict[metric_name] = one_metric_params
        return params_dict
    
    
    def set_params(self, params: dict) -> None:
        '''
        Вычисляет метрику с наибольшей дисперсией. Устанавливает параметры ее распределения как основные.

        args:
            params - словарь с параметрами,
            calc_metrics_df - датафрейм с рассчитанной ratio-метрикой для расчета среднего
        return:
            None
        '''
        stds = {key:val['std'] for key,val in params.items()}
        highest_variance_key = sorted(stds.items(), key=lambda x: x[1], reverse=True)[0][0]
        self.highest_variance_metric = highest_variance_key
        self.MEAN, self.VAR, self.STD = params[highest_variance_key]['mean'], params[highest_variance_key]['var'], params[highest_variance_key]['std']
        return


    def design_test(self, data: pd.DataFrame, strata_df: pd.DataFrame, exception_list: list, add_effect: float=None) -> None:
        '''

        '''
        start_time = datetime.now()
        remove_outliers_df = self.remove_outs(data, exception_list)[1]
        print('Start Time: ', start_time)
        print('    Проведена обработка выбросов')
        self.calc_metrics_df, _ = self.calc_metrics(
            remove_outliers_df, self.metric_dict, self.column_for_grouped, kappa=None, is_history_data=True)
        print('Рассчитаны метрики')
        self.params = self.calc_distribution_params(remove_outliers_df, strata_df, self.calc_metrics_df)
        self.set_params(self.params)
        print('    Установлены параметры распределения')

        self.sample_size_matrix = self.get_sample_size_matrix(
                self.get_sample_size_complex, self.MEAN, self.STD, plot=False)

        self.effects_matrix = self.get_effects_matrix(
                self.get_MDE, self.MEAN, self.STD, plot=False)

        self.data_distribution_plot([self.calc_metrics_df[self.target_metric_name]])
        display(self.sample_size_matrix)
        display(self.effects_matrix)
        
        
        eff_percent = self.sample_size_matrix.loc[0,'effects'] - 1
        if self.sample_size is None:
            self.sample_size = self.sample_size_matrix.loc[0,'sample_size']
        
        if add_effect:
            eff_percent += add_effect
        effect_size = np.ceil(np.mean(abs(self.calc_metrics_df[self.target_metric_name])) * eff_percent)
        print(f'Детектируем эффект в', round(eff_percent*100, 3), '% или' , effect_size, 'единиц в абсолютном выражении')
        print(f'    Размер выборки: {self.sample_size}')
        
        print('Запуск A/A/B теста...')
        pvalues_aa_res, pvalues_ab_res = self.aa_ab_test_history_calc(
            self.calc_metrics_df, self.sample_size, self.target_metric_name, 
            effect_size_abs=effect_size, test_type=self.simple_ttest
            )
        self.print_estimated_errors(pvalues_aa_res, pvalues_ab_res)
        self.plot_pvalue_distribution(pvalues_aa_res, pvalues_ab_res)
        
        print('Подбор контрольной и тестовой групп...')
        self.calc_metrics_strata_df = self.starta_merged(self.calc_metrics_df, strata_df, self.user_id_field_name)
        self.stratified_groups = self.get_stratified_groups(self.calc_metrics_strata_df, self.sample_size, weights=self.weights)
        time_elapsed = datetime.now() - start_time
        print('Done!')
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        return self.stratified_groups
    
    
    def test_estimation(self):
        pass