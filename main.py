import numpy as np
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

params = {'font.family': 'serif',
          'font.serif': 'FangSong',
          'font.style': 'italic',
          'font.weight': 'normal',  # or 'blod'
          'font.size': 12,  # 此处貌似不能用类似large、small、medium字符串
          'axes.unicode_minus': False
          }
rcParams.update(params)
import matplotlib.pyplot as plt
import pandas as pd
# 未来pandas版本会要求显式注册matplotlib的转换器，所以添加了下面两行代码，否则会报警告
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def load_data():
    """

    :rtype: object
    """
    from datetime import datetime
    date_parse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    data = pd.read_csv('KSF_Tea.csv',
                       index_col='date',  # 指定索引列
                       parse_dates=['date'],  # 将指定列按照日期格式来解析
                       date_parser=date_parse  # 日期格式解析器
                       )
    ts = data['Vol']
    print(ts.head(10))
    plt.plot(ts)
    plt.show()
    return ts


def use_rolling_statistics(time_series_datas):
    '''
    利用标准差和均值来肉眼观测时间序列数据的平稳情况
    :param time_series_datas:
    :return:
    '''
    roll_mean = time_series_datas.rolling(window=12).mean()
    roll_std = time_series_datas.rolling(window=12).std()
    # roll_variance = time_series_datas.rolling(window=12).var()
    plt.plot(time_series_datas, color='blue', label='Original')
    plt.plot(roll_mean, color='red', label='Rolling Mean')
    plt.plot(roll_std, color='green', label='Rolling Std')
    # plt.plot(roll_variance,color='yellow',label='Rolling Variance')

    plt.legend(loc='best')
    plt.title('利用Rolling Statistics来观测时间序列数据的平稳情况')
    plt.show(block=False)


def use_df(time_series_datas):
    '''
    迪基-富勒单位根检验
    :param time_series_datas:
    :return:
    '''
    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(time_series_datas, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def use_moving_avg(ts_log):
    moving_avg_month = ts_log.rolling(window=12).mean()
    plt.plot(moving_avg_month, color='green', label='moving_avg')
    plt.legend(loc='best')
    plt.title('利用移动平均法平滑ts_log序列')
    plt.show()
    return moving_avg_month


def use_exponentially_weighted_moving_avg(ts_log):
    expweighted_avg = ts_log.ewm(halflife=12).mean()
    plt.plot(expweighted_avg, color='green', label='expweighted_avg')
    plt.legend(loc='best')
    plt.title('利用指数加权移动平均法平滑ts_log序列')
    plt.show()
    return expweighted_avg


def use_decomposition(ts_log):
    '''
    时间序列分解
    :param ts_log:
    :return: 去除不平稳因素后的序列
    '''
    decomposition = seasonal_decompose(ts_log, freq=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    # 衡量趋势强度
    r_var = residual.var()
    tr_var = (trend + residual).var()
    f_t = np.maximum(0, 1.0 - r_var / tr_var)
    print(f_t)
    # 衡量季节性强度
    sr_var = (seasonal + residual).var()
    f_s = np.maximum(0, 1.0 - r_var / sr_var)
    print(f"-------趋势强度:{f_t},季节性强度:{f_s}------")
    return residual


def transform_stationary(ts):
    '''
    平稳变换：
    消除趋势：移动平均、指数加权移动平均
    有时候简单的减掉趋势的方法并不能得到平稳序列，尤其对于高季节性的时间序列来说，此时可以采用differencing(差分)或decomposition(分解)
    消除趋势和季节性：差分、序列分解
    :param ts:
    :return:
    '''
    # 利用log降低异方差性
    ts_log = np.log(ts)
    # plt.plot(ts_log, color='brown', label='ts_log')
    # plt.title('ts_log')
    # plt.show()

    # 移动平均法，得到趋势（需要确定合适的K值，当前例子中，合适的K值是12个月，因为趋势是逐年增长，但是有些复杂场景下，K值的确定很难）
    # trend = use_moving_avg(ts_log)
    # 指数加权移动平均法平，得到趋势(由于每次都是从当前时刻到起始时刻的指数加权平均，所以没有确定K值的问题)
    # trend = use_exponentially_weighted_moving_avg(ts_log)
    # print(trend)
    # 减去趋势：将平滑后的序列从ts_log序列中移除
    # rs = ts_log - trend
    # 若趋势建模是用的移动平均法，由于是取前12个月的均值，所以开始的11个值的移动平均都是非数了，需要去除非数
    # rs.dropna(inplace=True)

    # differencing(差分)
    rs_log_diff = ts_log - ts_log.shift()  # 1阶差分
    # use_rolling_statistics(rs)
    # rs = rs - rs.shift() # 2阶差分
    # 季节性差分 ,此案例中的季节间隔为12个月  d=1 D=1
    # rs = (ts_log - ts_log.shift(periods=12)) - (ts_log.shift() - ts_log.shift().shift(periods=12))
    rs_log_diff.dropna(inplace=True)

    # decomposition(分解)
    # rs = use_decomposition(ts_log)
    # rs.dropna(inplace=True)

    # 对去除趋势后的序列做平稳性检验
    # use_rolling_statistics(rs)
    use_df(rs_log_diff)
    return ts_log, rs_log_diff


def order_determination(ts_log_diff):
    '''
    利用acf和pacf确定模型以及阶数
    :param ts_log_diff:
    :return:
    '''
    lag_acf = acf(ts_log_diff, nlags=10, fft=False)
    lag_pacf = pacf(ts_log_diff, nlags=10, method='ols')
    z = 1.96
    # z = 1.65
    # Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-z / np.sqrt(len(ts_log_diff) - 1), linestyle='--',
                color='gray')  # 利用白噪声的标准正态分布假设来选择相关性的置信度区间，1.96是95%置信度下的统计量
    plt.axhline(y=z / np.sqrt(len(ts_log_diff) - 1), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')
    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-z / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=z / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()


def draw_rss_plot(ts_log_diff, orders, title, freq='MS'):
    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(ts_log_diff, order=orders, freq=freq)
    results_fitted = model.fit(disp=-1)
    # print(results.summary())
    plt.plot(ts_log_diff)
    plt.plot(results_fitted.fittedvalues, color='red')
    plt.title('%s RSS: %.4f' % (title, sum((results_fitted.fittedvalues - ts_log_diff) ** 2)))
    plt.show()
    return results_fitted.fittedvalues


def draw_future_plot(ts_log_diff, orders, seasonal_order, title, freq='MS'):
    # ARIMA模型
    # model = ARIMA(ts_log_diff, order=orders, freq=freq)
    # results_fitted = model.fit(disp=-1, trend='c')
    # fit_values = results_fitted.fittedvalues
    # fc, _, conf = results_fitted.forecast(36, alpha=0.05)  # 95% conf

    # 季节性ARIMA模型
    model = SARIMAX(ts_log_diff, order=orders, seasonal_order=seasonal_order)
    results_fitted = model.fit(disp=5)
    fit_values = results_fitted.fittedvalues
    print(results_fitted.summary())
    fc = results_fitted.forecast(36)
    conf = None

    return fit_values, fc, conf, title


def build_arima(ts_log_diff):
    '''
    start_params表示ARIMA模型的所有项的参数，包括常数项，AR阶数项，MA阶数项，随机误差项.
    '''
    # order = (0, 1, 0) # 仅能靠常数的逆差分构建一个趋势,这里的常数是start_params的第一个元素，是通过一个全一的exog列向量和一个endog列向量做OLS方法得到的一个常数，这个常数其实就是endog向量元素的平均值
    # order = (3, 1, 0) # 逆差分构建一个趋势 + 变量自回归拟合一定的波动
    # order = (0, 1, 3) # 逆差分构建一个趋势 + 随机误差自回归拟合一定的波动，误差应该是来自平均值作为预测的误差，待求证
    order = (3, 0, 2)  # 变量自回归拟合一定的波动 + 预测误差自回归拟合一定的波动
    seasonal_order = (0, 1, 0, 12)  # 季节性差分，季节窗口=12个月

    # draw_rss_plot(ts_log_diff, order, '拟合：%s' % str(order))

    fittedvalues, fc, conf, title = draw_future_plot(ts_log_diff, order, seasonal_order,
                                                     '预测：%s,%s' % (str(order), str(seasonal_order)))
    return fittedvalues, fc, conf, title


def transform_back(ts, fittedvalues, fc, conf, title):
    '''
    变换回平稳变换之前的状态，以便预测目标观测值
    :param ts: 原始序列
    :param fittedvalues: 拟合出的序列
    :param fc: 预测的未来序列
    :return:
    '''
    # Make as pandas series
    future_index = pd.date_range(start=ts.index[-1], freq='MS', periods=36)
    fc_series = pd.Series(fc, index=future_index)
    print(fc_series.head())
    print(fittedvalues.head(24))
    lower_series, upper_series = None, None
    if conf is not None:
        lower_series = pd.Series(conf[:, 0], index=future_index)
        upper_series = pd.Series(conf[:, 1], index=future_index)

    current_ARIMA_log = pd.Series(fittedvalues, copy=True)
    future_ARIMA_log = pd.Series(fc_series, copy=True)

    # 逆log
    current_ARIMA = np.exp(current_ARIMA_log)
    future_ARIMA = np.exp(future_ARIMA_log)
    # lower_ARIMA = np.exp(lower_log_series)
    # upper_ARIMA = np.exp(upper_log_series)
    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(ts, label='current_actual')
    plt.plot(current_ARIMA, label='current_fit')
    plt.plot(future_ARIMA, label='forecast', marker='o', ms=3)
    if lower_series is not None:
        # plt.fill_between(lower_ARIMA.index, lower_ARIMA, upper_ARIMA,color='k', alpha=.15)
        pass
    plt.title('Forecast vs Actuals %s' % title)
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


def plot_lag(rs):
    from pandas.plotting import lag_plot
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:4]):
        lag_plot(rs, lag=i + 1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i + 1))

    fig.suptitle('Lag Plots of AirPassengers', y=1.15)
    plt.show()


def SampEn(U, m, r):
    """
    Compute Sample entropy
    用于量化时间序列的可预测性
    思想：
    返回一个-np.log(A/B)，该值越小预测难度越小，所以A/B越大，预测难度越小。
    :param U: 时间序列
    :param m: 模板向量维数
    :param r: 距离容忍度，一般取0.1~0.25倍的时间序列标准差，也可以理解为相似度的度量阈值，小于这个阈值的2个向量被认为是相似的
    :return: 返回一个-np.log(A/B)，该值越小预测难度越小，所以A/B越大，预测难度越小。 一般可以和同等长度的随机序列的结果比较，小于这个结果，则具备一定的可预测性
    """

    def _maxdist(x_i, x_j):
        """
         Chebyshev distance
        :param x_i:
        :param x_j:
        :return:
        """
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))


if __name__ == '__main__':
    # 加载时间序列数据
    _ts = load_data()
    # 使用样本熵评估可预测性
    print(f'原序列样本熵:{SampEn(_ts.values, m=2, r=0.2 * np.std(_ts.values))}')
    # 检验平稳性
    use_rolling_statistics(_ts)  # rolling 肉眼
    use_df(_ts)  # Dickey-Fuller Test 量化
    # 平稳变换
    _ts_log, _rs_log_diff = transform_stationary(_ts)
    # 使用样本熵评估可预测性
    print(f'平稳变换后的序列样本熵:{SampEn(_ts.values, m=2, r=0.2 * np.std(_ts.values))}')
    # acf,pacf定阶分析
    order_determination(_rs_log_diff)
    # plot_lag(_rs)# lag plot(滞后图分析相关性)
    # 构建模型
    _fittedvalues, _fc, _conf, _title = build_arima(
        _ts_log)  # 这里只传取log后的序列是因为后面会通过指定ARIMA模型的参数d=1来做一阶差分，这样在预测的时候，就不需要手动做逆差分来还原序列，而是由ARIMA模型自动还原
    # 预测,并绘制预测结果图
    transform_back(_ts, _fittedvalues, _fc, _conf, _title)