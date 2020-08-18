# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batman = pd.read_csv('AAPL Historical Data.csv', usecols = [0,1,2,3,4])

POHL_avg = batman[['Price','Open','High','Low']].mean(axis = 1)


a = np.arange(1,len(batman)+1,1)




plt.plot(a,POHL_avg,'r',label = 'MY First Plot')
