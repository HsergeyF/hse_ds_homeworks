import math
import numpy as np
import pandas as pd
from scipy import stats
from statistics import mean, stdev


class MetricProvider:

    def __init__(self, rf):
        self.risk_free_rate = rf
    

    get_traditional_sharpe_ratio = staticmethod(lambda returns: returns.mean()/returns.std(ddof=1))
    get_traditional_sharpe_ratio.__doc__ = "Отношение среднего ретерна к стандартному отклонению ретернов"

    get_avg_return = staticmethod(lambda returns: returns.mean())
    get_traditional_sharpe_ratio.__doc__ = "Средний ретерн"
    
    get_returns_skew = staticmethod(lambda returns: stats.skew(returns))
    get_returns_skew.__doc__ = "Коэф. ассиметрии ретернов"

    get_returns_kurtosis = staticmethod(lambda returns: stats.kurtosis(returns, fisher=True))
    get_returns_skew.__doc__ = "Коэф. эксцесса ретернов"

    get_annualized_sharpe_ratio = staticmethod(lambda sharpe, days=261: sharpe * np.sqrt(days))
    get_annualized_sharpe_ratio.__doc__ = "Sharpe ratio аннуализированный к торговому году. На традиционных рынках в году 261 рабочий день."

    get_algo_sharpe_ratio = staticmethod(lambda returns, days=261: mean(returns) * days) / (stdev(returns) * math.sqrt(days))
    get_algo_sharpe_ratio.__doc__ = "Sharpe ratio для алгоритмических систем. На традиционных рынках в году 261 рабочий день."

    get_profitable_days_percent = staticmethod(lambda profits, returns: len(profits) / len(returns))
    get_profitable_days_percent.__doc__ = "Процент прибыльных ретернов от общей совокупности"

    get_profit_factor = staticmethod(lambda profits, losses: abs(sum(profits) / sum(losses)))
    get_profit_factor.__doc__ = "Абсолютное значение сумм прибылей деленное на сумму убытков"

    get_pnl = staticmethod(lambda equity: equity.loc[len(equity) - 1, 'strategyPl'])
    get_pnl.__doc__ = "Конечная доходность"

    get_recovery_factor =  staticmethod(lambda pl, max_drawdown: pl / max_drawdown)
    get_recovery_factor.__doc__ = "Конечная доходность деленная на максимальную просадку"

    get_profitable_returns = staticmethod(lambda returns: list(filter(lambda x: x > 0, returns)))
    get_profitable_returns.__doc__ = "Список положительных ретернов"

    get_lossing_returns = staticmethod(lambda returns: list(filter(lambda x: x < 0, returns)))
    get_profitable_returns.__doc__ = "Список отрицательных ретернов"
    
    @staticmethod
    def get_max_drawdown(returns):
        """
        Алгоритм поиска максимальной просадки в I(1) временном ряде.
        """
        local_maximum, current_equity, drawdown, drawdown_max = (0, 0, 0, 0)
        for value in returns:
            current_equity += value
            if local_maximum < current_equity:
                drawdown = 0
                local_maximum = current_equity
            else:
                drawdown += value
                if drawdown < drawdown_max:
                    drawdown_max = drawdown
        return abs(drawdown_max)
    
    @staticmethod
    def get_consequent_wins_losses(returns):
        max_cons_wins = 0
        """
        Рассчет максимального количество подряд идущих положительных и отрицательных ретернов.
        #TODO: рефактор в onliner - map()
        """
        max_cons_loss = 0
        cons_losses, cons_wins = [], []
        for value in returns:
            if value >= 0:
                cons_losses.append(max_cons_loss)
                max_cons_loss = 0
                max_cons_wins += 1
            else:
                cons_wins.append(max_cons_wins)
                max_cons_loss += 1
                max_cons_wins = 0
        return cons_losses, cons_wins
    
    def get_max_drawdown_percent(self, returns, pl):
        """
        Максимальная просадка в процентах
        """
        return self.get_max_drawdown(returns) / (max(pl) / 100 + 1)
    
    def get_calmar_ratio(self, pl, max_dd):
        """
        Доходность деленная на максимальную просадку
        """
        return abs((pl - self.risk_free_rate) / max_dd)
    
    def get_sortino_ratio(self, pl, losses):
        """
        Доходность деленная на стандартное отклонение отрицательных ретернов
        """
        return (pl - self.risk_free_rate) / stdev(losses)
    
    def get_p_value(self, returns, h0Mean=0):
        """
        P-value для нулевой гипотезы, что средний ретерн больше трешхолда,
        """
        mn = mean(returns)
        std = stdev(returns)
        zScore = (mn - h0Mean) / (std / math.sqrt(len(returns) - 1))
        cdf = stats.norm.cdf(zScore)
        return 1 - cdf
    
    def get_probabilistic_sharpe_ratio(self, returns=None, sr_benchmark=0.0):
        """
        Реализация вероятностного sharpe ratio из статьи Robust Portfolio Optimization 
        with VAR Adjusted Sharpe Ratio.
        Source: https://www.researchgate.net/figure/The-Probabilistic-Sharpe-Ratio-P-SRSR_fig2_256034137
        """
        sr = self.get_traditional_sharpe_ratio(returns)
        n = len(returns)
        skew = self.get_skew(returns)
        kurtosis = self.get_kurtosis(returns)

        sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))
        return stats.norm.cdf((sr - sr_benchmark) / sr_std)
    
    def get_deflated_sharpe_ratio(self, trials_returns=None, returns_selected=None, expected_mean_sr=0.0):
        """
        Реализация deflated sharpe ratio. Скорректированный коэфициент помогает лучше отфильтровывать false-positive
        стратегии по результатам бек-теста.
        Source: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
        """
        emc = 0.5772156649  # Euler-Mascheroni constant

        m = trials_returns.shape[1]
        corr_matrix = trials_returns.corr()
        p = corr_matrix.values[np.triu_indices_from(
            corr_matrix.values, 1)].mean()
        n = p + (1 - p) * m
        independent_trials = int(n) + 1

        srs = self.get_traditional_sharpe_ratio(trials_returns)
        trials_sr_std = srs.std()

        maxZ = (1 - emc) * stats.norm.ppf(1 - 1. / independent_trials) + emc * stats.norm.ppf(
            1 - 1. / (independent_trials * np.e))
        expected_max_sr = expected_mean_sr + (trials_sr_std * maxZ)
        return self.get_probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)
    
    def information_ratio(pnls):
        """
        Information Ratio включает в себя информацию о временном ряде underlying, на котором мы строили стратегию,
        рассчитывается как разница конечных результатов кумулятивной доходности стратегии и бенчмарка, деленная на 
        стандартное отклонение их разниц.
        """
        pnls['diff'] = pnls['strategy'] - pnls['benchmark']
        return (pnls.loc[pnls.index[-1], 'strategy'] -  pnls.loc[pnls.index[-1], 'benchmark'])/pnls['diff'].std()