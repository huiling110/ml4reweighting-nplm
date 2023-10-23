import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import math, os, h5py
import pandas as pd

import tensorflow as tf
import h5py
import os
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
from tensorflow import linalg as la
from tensorflow.keras import initializers
import numpy as np

#mean_REF = [1.34129594e+02,  9.34971789e+01,  4.73537438e-04, -1.22469242e-03, -1.12269829e-04,  2.43701885e+02]
#std_REF  = [317.04812272,    274.00141718,    1.00488891,     1.01146798,      2.44952327,       715.87708648]

# values used to standardize datasets (from 2018 only, central value REF distributions, weighted averges)
# keys   = ['leadmupt', 'subleadmupt', 'leadmueta', 'subleadmueta', 'delta_phi', 'mass']
mean_REF = [1.42025823e+02,  1.11705495e+02, -9.76846883e-04, -7.98511812e-04, -2.03257011e-03,  2.87895636e+02]
std_REF  = [4.03458111e+03, 2.55117576e+03, 1.04282100e+00, 1.04527193e+00, 8.34918949e+00, 1.32496084e+04]

trim_list_data16 = ['trim_Run2016B_SM_DM', 
                    'trim_Run2016C_SM_DM', 
                    'trim_Run2016D_SM_DM', 
                    'trim_Run2016E_SM_DM',
                    'trim_Run2016F_SM_DM',
                    'trim_Run2016G_SM_DM',
                    'trim_Run2016H_SM_DM']

trim_list_data17 = ['trim_Run2017B_SM_DM', 
                    'trim_Run2017C_SM_DM', 
                    'trim_Run2017D_SM_DM', 
                    'trim_Run2017E_SM_DM',
                    'trim_Run2017F_SM_DM']

trim_list_data18 = ['trim_Run2018A_SM_DM', 
                    'trim_Run2018B_SM_DM', 
                    'trim_Run2018C_SM_DM', 
                    'trim_Run2018D_SM_DM']
trim_list_mc16 = [
             'trim_WJetsToLNu', 'trim_WW_plus_ext1', 'trim_WZTo2L2Q', 'trim_WZTo3LNu', 
             'trim_ZZTo2L2Q', 'trim_ZZTo2L2Nu', 'trim_ZZTo4L',
             'trim_ZToMuMu_M_50_120_theoryUnc_57files', 'trim_ZToMuMu_M_120_200_ext1_theoryUnc', 'trim_ZToMuMu_M_200_400_ext1_theoryUnc', 'trim_ZToMuMu_M_400_800_ext1_theoryUnc',
             'trim_ZToMuMu_M_800_1400_theoryUnc', 'trim_ZToMuMu_M_1400_2300_theoryUnc', 'trim_ZToMuMu_M_2300_3500_theoryUnc_2files', 'trim_ZToMuMu_M_3500_4500_theoryUnc', 
             'trim_ZToMuMu_M_4500_6000_theoryUnc', 'trim_ZToMuMu_M_6000_Inf_theoryUnc', 
             'trim_TTTo2L2Nu', 'trim_TTToSemilepton', 'trim_ST_tW_top_5f', 'trim_ST_tW_antitop_5f',
             'trim_ST_tchannel_top_4f', 'trim_ST_tchannel_antitop_4f', 
            ]

trim_list_mc17 = [
             'trim_WJetsToLNu_plus_ext1'  , 'trim_WZTo2L2Q'              , 'trim_WZTo3LNu'    , 'trim_WWTo2L2Nu_PSweights' ,
             'trim_ZZTo2L2Q'              , 'trim_ZZTo2L2Nu'             , 'trim_ZZTo4L_1star',
             'trim_ZToMuMu_M_50_120_52files', 'trim_genmass_ZToMuMu_M_120_200_ext1', 'trim_genmass_ZToMuMu_M_200_400_ext1', 'trim_genmass_ZToMuMu_M_400_800_ext1'     ,
             'trim_genmass_ZToMuMu_M_800_1400'    , 'trim_genmass_ZToMuMu_M_1400_2300'   , 'trim_genmass_ZToMuMu_M_2300_3500', 'trim_genmass_ZToMuMu_M_3500_4500'   , 
             'trim_genmass_ZToMuMu_M_4500_6000'   , 'trim_genmass_ZToMuMu_M_6000_Inf'    , 
             'trim_TTTo2L2Nu_PSweights'       , 'trim_TTToSemiLeptonic_PSweights_1star', 
             'trim_ST_tW_top_5f_PSweights', 'trim_ST_tW_antitop_5f_PSweights'      ,
             'trim_ST_tchannel_top_5f'    , 'trim_ST_tchannel_antitop_5f_PSweights', 'trim_ST_schannel_4f_PSweights', 
            ]
trim_list_mc18 = [
             'trim_WJetsToLNu'            , 'trim_WW'                    , 'trim_WZTo2L2Q'             , 'trim_WZTo3LNu'             , 
             'trim_ZZTo2L2Q'              , 'trim_ZZTo2L2Nu'             , 'trim_ZZTo4L'               ,
             'trim_ZToMuMu_M_50_120' ,    'trim_ZToMuMu_M_120_200_ext1', 'trim_ZToMuMu_M_200_400_ext1', 'trim_ZToMuMu_M_400_800_ext1' ,
             'trim_ZToMuMu_M_800_1400'    , 'trim_ZToMuMu_M_1400_2300'   , 'trim_ZToMuMu_M_2300_3500'  , 'trim_ZToMuMu_M_3500_4500'   , 
             'trim_ZToMuMu_M_4500_6000'   , 'trim_ZToMuMu_M_6000_Inf'    , 
             'trim_TTTo2L2Nu_1star'       , 'trim_TTToSemiLeptonic_1star', 
             'trim_ST_tW_top_5f'          , 'trim_ST_tW_antitop_5f'      ,
             'trim_ST_tchannel_top_5f'    , 'trim_ST_tchannel_antitop_5f', 'trim_ST_schannel_4f', 
            ]
LUMINOSITY16 = 0
LUMINOSITY16+= 5746.01 #/pb for Run2016B
LUMINOSITY16+= 2572.52 #/pb for Run2016C
LUMINOSITY16+= 4242.29 #/pb for Run2016D
LUMINOSITY16+= 4025.22 #/pb for Run2016E
LUMINOSITY16+= 3104.51 #/pb for Run2016F
LUMINOSITY16+= 7575.58 #/pb for Run2016G
LUMINOSITY16+= 8650.63 #/pb for Run2016H

LUMINOSITY17 = 0
LUMINOSITY17+=4793.960 #/pb for Run2017B
LUMINOSITY17+=9631.610 #/pb for Run2017C
LUMINOSITY17+=4247.680 #/pb for Run2017D
LUMINOSITY17+=9313.640 #/pb for Run2017E
LUMINOSITY17+=13539.04 #/pb for Run2017F

LUMINOSITY18  = 0
LUMINOSITY18 +=13977.334 #/pb Run2018A
LUMINOSITY18 +=7057.8000 #/pb Run2018B
LUMINOSITY18 +=6894.8000 #/pb Run2018C
LUMINOSITY18 +=31742.600 #/pb Run2018D

xsec_dict16  = { 'trim_DYJetsToLL_M50'        : 6225., 
                 'trim_WJetsToLNu'            : 61526.7, 
                 'trim_WW_plus_ext1'          : 115.,
                 'trim_WZTo2L2Q'              : 6.331,
                 'trim_WZTo3LNu'              : 5.052, 
                 'trim_ZZTo2L2Q'              : 3.688, 
                 'trim_ZZTo2L2Nu'             : 0.5644, 
                 'trim_ZZTo4L'                : 1.325, 
                 'trim_TTTo2L2Nu'             : 88.29,
                 'trim_TTToSemilepton'        : 365.34, 
                 'trim_ST_tW_top_5f'          : 19.2,
                 'trim_ST_tW_antitop_5f'      : 19.23, 
                 'trim_ST_tchannel_top_4f'    : 119.7,
                 'trim_ST_tchannel_antitop_4f': 71.74, 
                 'trim_ST_schannel_4f'        : 3.74,
                
                 'trim_ZToMuMu_M_50_120_theoryUnc_57files': 1975,
                 'trim_ZToMuMu_M_120_200_ext1_theoryUnc': 19.32,
                 'trim_ZToMuMu_M_200_400_ext1_theoryUnc': 2.731,
                 'trim_ZToMuMu_M_400_800_ext1_theoryUnc': 0.241,
                 'trim_ZToMuMu_M_800_1400_theoryUnc'    : 0.01678,
                 'trim_ZToMuMu_M_1400_2300_theoryUnc'   : 0.00139,
                 'trim_ZToMuMu_M_2300_3500_theoryUnc_2files': 0.00008948,
                 'trim_ZToMuMu_M_3500_4500_theoryUnc'   : 0.000004135,
                 'trim_ZToMuMu_M_4500_6000_theoryUnc'   : 0.000000456,
                 'trim_ZToMuMu_M_6000_Inf_theoryUnc'    : 0.0000000206
               } # pb

xsec_dict17  = { 'trim_DYJetsToLL_M50'        : 6225., 
                 'trim_WJetsToLNu_plus_ext1'  : 61526.7, 
                 'trim_WWTo2L2Nu_PSweights'   : 10.48,
                 'trim_WZTo2L2Q'              : 6.331,
                 'trim_WZTo3LNu'              : 5.052, 
                 'trim_ZZTo2L2Q'              : 3.688, 
                 'trim_ZZTo2L2Nu'             : 0.5644, 
                 'trim_ZZTo4L_1star'          : 1.325, 
                 'trim_TTTo2L2Nu_PSweights'   : 88.29,
                 'trim_TTToSemiLeptonic_PSweights_1star': 365.34, 
                 'trim_ST_tW_top_5f_PSweights'          : 19.2,
                 'trim_ST_tW_antitop_5f_PSweights'      : 19.23, 
                 'trim_ST_tchannel_top_5f'    : 119.7,
                 'trim_ST_tchannel_antitop_5f_PSweights': 71.74, 
                 'trim_ST_schannel_4f_PSweights': 3.74,
                 'trim_ZToMuMu_M_50_120_52files': 2112.904,
                 'trim_ZToMuMu_M_120_200_ext1'  : 20.553,
                 'trim_ZToMuMu_M_200_400_ext1'  : 2.886,
                 'trim_ZToMuMu_M_400_800_ext1'  : 0.2517,
                 'trim_ZToMuMu_M_800_1400'    : 0.01707,
                 'trim_ZToMuMu_M_1400_2300'   : 0.001366,
                 'trim_ZToMuMu_M_2300_3500'   : 0.00008178,
                 'trim_ZToMuMu_M_3500_4500'   : 0.000003191,
                 'trim_ZToMuMu_M_4500_6000'   : 0.0000002787,
                 'trim_ZToMuMu_M_6000_Inf'    : 0.000000009569
               } # pb

xsec_dict18  = { 'trim_DYJetsToLL_M50'        : 6225., 
                 'trim_WJetsToLNu'            : 61526.7, 
                 'trim_WW'                    : 115.,
                 'trim_WZTo2L2Q'              : 6.331,
                 'trim_WZTo3LNu'              : 5.052, 
                 'trim_ZZTo2L2Q'              : 3.688, 
                 'trim_ZZTo2L2Nu'             : 0.5644, 
                 'trim_ZZTo4L'                : 1.325, 
                 'trim_TTTo2L2Nu_1star'       : 88.29,
                 'trim_TTToSemiLeptonic_1star': 365.34, 
                 'trim_ST_tW_top_5f'          : 19.2,
                 'trim_ST_tW_antitop_5f'      : 19.23, 
                 'trim_ST_tchannel_top_5f'    : 119.7,
                 'trim_ST_tchannel_antitop_5f': 71.74, 
                 'trim_ST_schannel_4f'        : 3.74,
                 'trim_ZToMuMu_M_50_120'      : 2112.904,
                 'trim_ZToMuMu_M_120_200_ext1': 20.553,
                 'trim_ZToMuMu_M_200_400_ext1': 2.886,
                 'trim_ZToMuMu_M_400_800_ext1': 0.2517,
                 'trim_ZToMuMu_M_800_1400'    : 0.01707,
                 'trim_ZToMuMu_M_1400_2300'   : 0.001366,
                 'trim_ZToMuMu_M_2300_3500'   : 0.00008178,
                 'trim_ZToMuMu_M_3500_4500'   : 0.000003191,
                 'trim_ZToMuMu_M_4500_6000'   : 0.0000002787,
                 'trim_ZToMuMu_M_6000_Inf'    : 0.000000009569
               } # pb
trim_list_data = ['trim_Run2018A_SM_DM', 'trim_Run2018B_SM_DM', 'trim_Run2018C_SM_DM', 'trim_Run2018D_SM_DM']
columns_MC   = ['mcweight', 'puweight', 'exweight', 'trgweight', 
               'm1dB', 'm1dz', 'm1iso', 'm1pt', 'm1eta', 'm1phi', 'm1SF', 'm1SFErr',
               'm2dB', 'm2dz', 'm2iso', 'm2pt', 'm2eta', 'm2phi', 'm2SF', 'm2SFErr',
               'mass', 'dimuonpt', 'nbjets', 'nmu', 'genleadmupt'
              ]
columns_scalecorr = ['m1SF', 'm1SFErr', 'm2SF', 'm2SFErr']
columns_training  = ['leadmupt', 'subleadmupt', 'leadmueta', 'subleadmueta', 'delta_phi', 'mass']
columns_weight    = ['weight']

def DeltaPhi(phi1, phi2):
    result  = phi1 - phi2;
    result -= 2*math.pi*(result >  math.pi)
    result += 2*math.pi*(result <= -math.pi)
    return result

def Apply_MuonMomentumScale_Correction(data_5D, scale_4D, muon_scale=0):
    muon_mass = 0.1#0565837 #GeV/c2  
    m1SF = scale_4D[:, 0]
    m2SF = scale_4D[:, 2]
    m1SF_err = scale_4D[:, 1]
    m2SF_err = scale_4D[:, 3]
    
    pt1  = data_5D[:, 0]/m1SF*(m1SF + muon_scale*m1SF_err)
    pt2  = data_5D[:, 1]/m2SF*(m2SF + muon_scale*m2SF_err)
    eta1 = data_5D[:, 2]
    eta2 = data_5D[:, 3]
    dphi = data_5D[:, 4]

    px1= pt1
    px2= pt2*np.cos(dphi)
    py1= np.zeros_like(pt1)
    py2= pt2*np.sin(dphi)
    pz1= pt1*np.sinh(eta1)
    pz2= pt2*np.sinh(eta2)
    E1 = np.sqrt(px1*px1+py1*py1+pz1*pz1+muon_mass*muon_mass)
    E2 = np.sqrt(px2*px2+py2*py2+pz2*pz2+muon_mass*muon_mass)

    px = px1+px2
    py = py1+py2
    pz = pz1+pz2
    E  = E1+E2
    mll= np.sqrt(E*E-px*px-py*py-pz*pz)

    data_5D_new      = np.copy(data_5D)
    data_5D_new[:, 0]= pt1
    data_5D_new[:, 1]= pt2
    data_5D_new[:, 5]= mll
    return data_5D_new

def read_data_training_nu(folder, year, mass_cut, nu, muonpt_scale_str, trim_list, columns_training, columns_weight):
    print(muonpt_scale_str)
    mc_folder   = '%s/MC_%s_M%i_final_match_muonPTscale_%s/'%(folder, year, mass_cut, muonpt_scale_str)
    DATA   = np.array([])
    W_DATA = np.array([])
    Y_DATA = np.array([])
    REF    = np.array([])
    W_REF  = np.array([])
    i=0
    for process in trim_list:
        if 'DY' in process: continue
        f = h5py.File(mc_folder+process+'.h5', 'r')
        read_file = np.array([])
        for p in columns_training+columns_weight:
            col = np.array(f.get(p))
            col = np.expand_dims(col, axis=1)
            if read_file.shape[0]==0:
                read_file = col
            else:
                read_file = np.concatenate((read_file, col), axis=1)
        if REF.shape[0]==0:
            REF    = read_file[:, :-1]
            W_REF  = read_file[:, -1:]
        else:
            REF    = np.concatenate((REF,  read_file[:, :-1]), axis=0)
            W_REF  = np.concatenate((W_REF,  read_file[:, -1:]), axis=0)
        i+=1
    mask = (W_REF[:, 0]>0)*(W_REF[:, 0]<0.5)*(REF[:, 0]>0)*(REF[:, 1]>0)
    return REF[mask], W_REF[mask]

def random_pick(feature, target, seed, fraction=0.1):
    np.random.seed(seed)
    N = feature.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    mask = (idx<int(N*fraction))
    return tf.convert_to_tensor(feature[mask], dtype=tf.float32), tf.convert_to_tensor(target[mask], dtype=tf.float32)

### Model
def ParametricLinearLoss_c(true, pred):
    a1 = pred[:, 0]
    y  = true[:, 0]
    w  = true[:, 1]
    nu = true[:, 2]
    f  = tf.multiply(a1, nu)
    c  = 1./(1+tf.exp(f))
    return tf.reduce_mean(y*w*c**2 + (1-y)*w*(1-c)**2)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                                                                                
    '''
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return tf.clip_by_value(p, clip_value_min=-self.c, clip_value_max=self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}
    
class BSMfinder(Model):
    def __init__(self,input_shape, architecture=[1, 4, 1], weight_clipping=1.0, activation='sigmoid', trainable=True, initializer=None, name=None, **kwargs):
        kernel_initializer="glorot_uniform"
        bias_initializer="zeros"
        if not initializer==None:
            kernel_initializer = initializer
            bias_initializer = initializer
        super().__init__(name=name, **kwargs)
        kernel_constraint = None
        if not weight_clipping==None: kernel_constraint = WeightClip(weight_clipping)
        self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation, trainable=trainable,
                                    kernel_constraint=kernel_constraint, kernel_initializer=initializer, bias_initializer=initializer) for i in range(len(architecture)-2)]
        self.output_layer  = Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear', trainable=trainable,
                                     kernel_constraint=kernel_constraint, kernel_initializer=initializer, bias_initializer=initializer)
        self.build(input_shape)
        
    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x

class BSMfinder_c(Model):
    def __init__(self,input_shape, architecture=[1, 4, 1], activation='sigmoid', l2=None, trainable=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        kernel_regularizer = None
        bias_regularizer   = None
        if not l2==None:
            kernel_regularizer = tf.keras.regularizers.L2(l2=l2, **kwargs)
            bias_regularizer = tf.keras.regularizers.L2(l2=l2, **kwargs)
        self.hidden_layers = [Dense(architecture[i+1], input_shape=(architecture[i],), activation=activation, 
                                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                    trainable=trainable) for i in range(len(architecture)-2)]
        self.output_layer  =  Dense(architecture[-1], input_shape=(architecture[-2],), activation='linear', 
                                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                    trainable=trainable)
        self.build(input_shape)

    def call(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x

    
class ParametricNet(Model):
    def __init__(self, input_shape, architecture=[1, 10, 1], activation='sigmoid', l2=None, poly_degree=1,
                 initial_model=None, train_coeffs=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.poly_degree = poly_degree
        if not isinstance(train_coeffs, list): self.train_coeffs = [train_coeffs for _ in range(self.poly_degree)]
        else: self.train_coeffs = train_coeffs
        
        self.coeffs = [
            BSMfinder_c(input_shape, architecture, activation=activation, trainable=self.train_coeffs[i]) for i in range(self.poly_degree) 
        ]
        self.build(input_shape)
        if not initial_model == None:
            self.load_weights(initial_model, by_name=True)

    def call(self, x):
        out = []
        for i in range(self.poly_degree):
            out.append(self.coeffs[i](x))
        if self.poly_degree == 1:
            return out[0]
        else:
            return tf.keras.layers.Concatenate(axis=1)(out)


def ParametricLoss_c(true, pred):
    f = pred[:, 0]
    y = true[:, 0]
    w = true[:, 1]
    c = 1./(1+tf.exp(f))
    return 100*tf.reduce_mean(y*w*c**2 + (1-y)*w*(1-c)**2)

def ParametricLoss_poly(true, pred):
    y = true[:, 0]
    w = true[:, 1]
    nu= true[:, 2]
    f = tf.zeros_like(y)
    for i in range(pred.shape[1]):
        f += pred[:, i]*(nu**(i+1))
    c = 1./(1+tf.exp(f))
    return 100*tf.reduce_mean(y*w*c**2 + (1-y)*w*(1-c)**2)

def Delta_poly(true, pred):
    y = true[:, 0]
    w = true[:, 1]
    nu= true[:, 2]
    f = tf.zeros_like(y)
    for i in range(pred.shape[1]):
        f += pred[:, i]*(nu**(i+1))
    return f.numpy()

### plots
labels_dict  = { 'trim_DYJetsToLL_M50'        : 5, 
                 'trim_WJetsToLNu'            : 0, 
                 'trim_WJetsToLNu_plus_ext1'  : 0,
                 'trim_WW'                    : 1,
                 'trim_WWTo2L2Nu_PSweights'   : 1,
                 'trim_WW_plus_ext1'          : 1,
                 'trim_WZTo2L2Q'              : 1,
                 'trim_WZTo3LNu'              : 1, 
                 'trim_ZZTo2L2Q'              : 1, 
                 'trim_ZZTo2L2Nu'             : 1, 
                 'trim_ZZTo4L'                : 1, 
                 'trim_ZZTo4L_1star'          : 1, 
                 'trim_TTTo2L2Nu_1star'       : 2,
                 'trim_TTTo2L2Nu'             : 2,
                 'trim_TTTo2L2Nu_PSweights'   : 2,
                 'trim_TTToSemiLeptonic_1star': 2,
                 'trim_TTToSemiLeptonic_PSweights_1star':2,
                 'trim_TTToSemilepton'        : 2, 
                 'trim_ST_tW_top_5f'          : 3,
                 'trim_ST_tW_top_5f_PSweights': 3,
                 'trim_ST_tW_antitop_5f'      : 3,
                 'trim_ST_tW_antitop_5f_PSweights': 3,
                 'trim_ST_tchannel_top_5f'    : 3,
                 'trim_ST_tchannel_top_4f'    : 3,
                 'trim_ST_tchannel_antitop_5f': 3, 
                 'trim_ST_tchannel_antitop_4f': 3,
                 'trim_ST_tchannel_antitop_5f_PSweights': 3,
                 'trim_ST_schannel_4f'        : 3,
                 'trim_ST_schannel_4f_PSweights' :3,
                 'trim_ZToMuMu_M_50_120'      : 4,
                 'trim_ZToMuMu_M_50_120_52files': 4,
                 'trim_ZToMuMu_M_50_120_theoryUnc_57files': 4,
                 'trim_ZToMuMu_M_120_200'     : 4,
                 'trim_ZToMuMu_M_120_200_ext1'     : 4,
                 'trim_ZToMuMu_M_120_200_theoryUnc': 4,
                 'trim_ZToMuMu_M_200_400'     : 4,
                 'trim_ZToMuMu_M_200_400_ext1'     : 4,
                 'trim_ZToMuMu_M_200_400_theoryUnc': 4,
                 'trim_ZToMuMu_M_400_800'     : 4,
                 'trim_ZToMuMu_M_400_800_ext1'     : 4,
                 'trim_ZToMuMu_M_400_800_theoryUnc': 4,
                 'trim_ZToMuMu_M_800_1400'    : 4,
                 'trim_ZToMuMu_M_800_1400_theoryUnc': 4,
                 'trim_ZToMuMu_M_1400_2300'   : 4,
                 'trim_ZToMuMu_M_1400_2300_theoryUnc': 4,
                 'trim_ZToMuMu_M_2300_3500'   : 4,
                 'trim_ZToMuMu_M_2300_3500_theoryUnc_2files': 4,
                 'trim_ZToMuMu_M_3500_4500'   : 4,
                 'trim_ZToMuMu_M_3500_4500_theoryUnc': 4,
                 'trim_ZToMuMu_M_4500_6000'   : 4,
                 'trim_ZToMuMu_M_4500_6000_theoryUnc': 4,
                 'trim_ZToMuMu_M_6000_Inf'    : 4,
                 'trim_ZToMuMu_M_6000_Inf_theoryUnc': 4
               } 
ref_labels = ['W+jets', 'WW+WZ+ZZ', r'$t\bar{t}$', r'$t/\bar{t}$', 'DY', 'DY']

mass_max = 6000
mass_min = 200
bins_dict = {
    'leadmupt': np.append(np.append(np.arange(0, 400, 20), np.arange(460, 700, 80)), [800, 900, 1200, 1900]),
    'subleadmupt': np.append(np.arange(0, 450, 50), [ 480, 600, 800, 1000, 1900, 3000]),
    'leadmueta': np.arange(-2, 2.1, 0.1),
    'subleadmueta': np.arange(-2, 2.1, 0.1),
    'delta_phi': np.arange(-3.5, 3.7, 0.2),
    'mass': np.append(np.arange(mass_min, 1100, 50), [1120, 1200,1300, 1450,1600, 1800, 2100, 2400, 3000])
}
xlabel_dict = {
    'leadmupt': r'$p_{\rm{T},1}$ (GeV/$c$)',
    'subleadmupt': r'$p_{\rm{T},2}$ (GeV/$c$)',
    'leadmueta': r'$\eta_1$',
    'subleadmueta': r'$\eta_2$',
    'delta_phi': r'$\Delta\phi_{1,2}$',
    'mass': r'$m_{\mu^{+}\mu^{-}}$ (GeV/$c^2$)'
}
colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3']
colors = ['midnightblue', '#1f78b4', 
          'lightseagreen', 'mediumseagreen', 
          'darkseagreen']
year_label_dict = {
    '2016': 0,
    '2017': 1,
    '2018': 2,
}

trim_list = {
    '2016': trim_list_mc16,
    '2017': trim_list_mc17,
    '2018': trim_list_mc18
}
