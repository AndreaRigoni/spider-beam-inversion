#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from strike_imgutils import *


# WITE FILES 1 GAUSS
#
gen = UniGen(dataset_size=100000, n_gauss=1)
gen.read_from_generator()
gen.write_to_files('uni1')
exit()






