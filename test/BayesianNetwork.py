!pip install pgmpy


import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

#임의로 지정
cpd_S = TabularCPD('S', 2, [[0.4], [0.6]])  
cpd_C = TabularCPD('C', 3, [[0.4], [0.3], [0.3]])  
cpd_E = TabularCPD('E', 3, [[0.4], [0.3], [0.3]])  

#확률 없어서 임의의로 지정했습니다.
cpd_L_on_SCE = TabularCPD('L', 2, np.array([[0.9, 0.7, 0.6, 0.8, 0.5, 0.4, 0.7, 0.6, 0.5, 0.3, 0.4, 0.2, 0.5, 0.3, 0.2, 0.4, 0.3, 0.2],[0.1, 0.3, 0.4, 0.2, 0.5, 0.6, 0.3, 0.4, 0.5, 0.7, 0.6, 0.8, 0.5, 0.7, 0.8, 0.6, 0.7, 0.8]]),evidence=['S', 'C', 'E'], evidence_card=[2, 3, 3])

# S,E,C = 0: 많음 1:보통 2:적음 L = 0: 적절 1: 부적절
print(cpd_L_on_SCE)


#베이지안 네트워크에 모델 추가
model = BayesianNetwork([('S', 'L'), ('E', 'L'), ('C', 'L')])

model.add_cpds(cpd_S, cpd_C, cpd_E, cpd_L_on_SCE)

#모델 검사
model.check_model()
