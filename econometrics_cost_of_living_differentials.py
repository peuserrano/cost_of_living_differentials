import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class AnaliseCustoVida:
    def __init__(self, file_path):
        self.base_pof = pd.read_csv(file_path, low_memory=False, encoding='latin1').dropna()
        self.base_dummies = self.create_dummy_variables()

    def create_dummy_variables(self):
        base_dummies = self.base_pof.copy()
        unique_values_controle = base_dummies['group'].unique()
        unique_values_regional = base_dummies['rm'].unique()

        for value in unique_values_controle:
            base_dummies[value] = (base_dummies['group'] == value).astype(int)

        for value in unique_values_regional:
            base_dummies[value] = (base_dummies['rm'] == value).astype(int)

        return base_dummies

    def run_regression(self):
        dummies_controle = self.base_dummies[['alimentacao', 'habitacao', 'vestuario', 'despesas diversas',
                                              'transporte', 'saude', 'higiene', 'servicos pessoasis',
                                              'recreacao e cultura', 'fumo', 'educacao']]
        dummies_regionais = self.base_dummies[['spa', 'for', 'rho', 'cur', 'rec', 'poa', 'rio', 'df', 'goi', 'sal', 'bel']]

        self.base_pof['log_rendapc'] = np.log(self.base_pof['rendapc'])
        self.base_pof['log_gastos_familiar_pc'] = np.log(self.base_pof['gastos_familiar_pc'])

        variaveis_explicativas = pd.merge(dummies_regionais, dummies_controle, left_index=True, right_index=True)
        variaveis_explicativas = variaveis_explicativas.drop(['rec', 'alimentacao'], axis=1)

        Y = self.base_pof['log_gastos_familiar_pc']
        X = variaveis_explicativas
        X = sm.add_constant(X)

        self.model = sm.OLS(Y, X).fit()

    def plot_results(self):
        vetor_parametros = pd.DataFrame(self.model.params)
        vetor_parametros['β^'] = vetor_parametros[0]
        vetor_parametros = vetor_parametros.drop(0, axis=1)
        vetor_parametros = vetor_parametros.T

        fig, ax = plt.subplots()

        diferenciais_estimados = (np.exp(vetor_parametros)) - 1
        diferenciais_regionais = (diferenciais_estimados)[['spa', 'for', 'rho', 'cur', 'poa', 'rio', 'df', 'goi', 'sal', 'bel']]

        cores = ['green' if beta > 0 else 'red' for beta in diferenciais_regionais.T['β^']]

        ax.bar(range(len(diferenciais_regionais.T['β^'])), diferenciais_regionais.T['β^'], color=cores)

        ax.set_xticks(range(len(diferenciais_regionais.T['β^'])))
        ax.set_xticklabels(list(diferenciais_regionais.T.index))

        ax.set_ylabel('Custo de vida em relação à RMR', fontweight='bold', labelpad=12)
        ax.set_title('Análise dos Diferenciais Regionais de Custo de Vida', fontweight='bold')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.show()

    def display_diferenciais(self):
        diferenciais_estimados = (np.exp(self.model.params)) - 1
        diferenciais_regionais = diferenciais_estimados[['spa', 'for', 'rho', 'cur', 'poa', 'rio', 'df', 'goi', 'sal', 'bel']]

        print('Diferencial de custos de vida (%) das regiões metropolitanas em relação à de Recife:')
        print(round(diferenciais_regionais * 100, 2))

# Utilização da classe
analise = AnaliseCustoVida("base_POF.csv")
analise.run_regression()
analise.plot_results()
analise.display_diferenciais()
