import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Настройка
np.random.seed(42)
n = 30
k = 1000

# Генерация выборочных средних
means_uniform = [np.mean(np.random.uniform(0, 1, size=n)) for _ in range(k)]
means_exp     = [np.mean(np.random.exponential(1.0, size=n)) for _ in range(k)]
means_poisson = [np.mean(np.random.poisson(3, size=n)) for _ in range(k)]

# === ГИСТОГРАММЫ === #
def plot_histogram(data, mu, sigma, title, filename):
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=False, stat='density', bins=30, color='skyblue', label='Эмпирическая гистограмма')
    
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, stats.norm.pdf(x, loc=mu, scale=sigma), 'r--', label='Теоретическая N(μ, σ²/n)')
    
    plt.title(title)
    plt.xlabel('Выборочное среднее')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Расчёт теоретических параметров
plot_histogram(means_uniform, mu=0.5, sigma=np.sqrt(1/12)/np.sqrt(n),
               title='Гистограмма средних для U(0,1)',
               filename='uniform_mean_hist.png')

plot_histogram(means_exp, mu=1.0, sigma=1.0/np.sqrt(n),
               title='Гистограмма средних для Exp(1)',
               filename='exp_mean_hist.png')

plot_histogram(means_poisson, mu=3.0, sigma=np.sqrt(3)/np.sqrt(n),
               title='Гистограмма средних для Pois(3)',
               filename='poisson_mean_hist.png')

# === QQ-ПЛОТЫ === #
def plot_qq(data, title, filename):
    plt.figure(figsize=(6, 6))
    sm.qqplot(np.array(data), line='45', fit=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_qq(means_uniform, title='QQ-плот для U(0,1)', filename='uniform_qq.png')
plot_qq(means_exp, title='QQ-плот для Exp(1)', filename='exp_qq.png')
plot_qq(means_poisson, title='QQ-плот для Pois(3)', filename='poisson_qq.png')

