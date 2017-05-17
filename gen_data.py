# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:25:50 2017

@author: Dsleviadm
"""

import numpy as np
import pandas as pd

n_trt=2
num_trt=1000
a=np.random(0, 1, size=[50, 1])
b=np.random(0, 1, size=[50, 1])
c=np.random(0, 1, size=[50, 1])


data_x=np.random.uniform(0, 10, size=[num_trt, 49])
data_x=np.append(data_x, np.random.uniform(0, 7.5, size=[num_trt, 1]), 1)
data_y=sum()

data=np.random.uniform(0, 10, size=[num_trt, 49])
data=np.append(data, np.random.uniform(2.5, 10, size=[num_trt, 1]), 1)