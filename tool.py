#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 22:03:09 2022

@author: xiaohaotian
"""

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import json
import timeit
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
# from scipy.signal import savgol_filter
# from scipy import interpolate

def get_data(dtype=np.double):
    L = 4591  # cm, longueur du pont
    Lmax = 6000 - 4591
    NI1 = L
    NI2 = L + Lmax
    y_c = 220
    e_R = 18

    y_g = y_c - 10 * int(e_R / 2)
    # ordonnée de la roue gauche en dm (par rapport à l'axe du tablier).
    # C'est un entier
    nyg = int((y_g / 10) + 44)
    # ordonnée de la roue droite en dm (par rapport à l'axe du tablier).
    # C'est un entier
    nyd = nyg + e_R
    # vitesse en m/s ou cm/cs (c'est un entier)


    s1 = "./Complet_Isym_ttcapt_V36.json"
    # Importons toutes ces fonctions d'influence dans le fichier data1
    f = open(s1)
    data1 = json.load(f)
    f.close()

    capteur1 = 'CO_T2_P2'
    capteur1S = 'CO_T2_P9'
    capteur2 = 'CO_T2_P4'
    capteur2S = 'CO_T2_P7'
    capteur3 = 'CO_T2_P7'
    capteur3S = 'CO_T2_P4'
    capteur4 = 'CO_T2_P9'
    capteur4S = 'CO_T2_P2'
    capteur5 = 'CO_T3_P4'
    capteur5S = 'CO_T3_P7'
    capteur6 = 'CO_T3_P7'
    capteur6S = 'CO_T3_P4'
    capteur7 = 'CO_AXL_23'
    capteur7S = 'CO_AXL_89'
    capteur8 = 'CO_AXL_45'
    capteur8S = 'CO_AXL_67'
    capteur9 = 'CO_AXL_67'
    capteur9S = 'CO_AXL_45'
    capteur10 = 'CO_AXL_89'
    capteur10S = 'CO_AXL_23'

    # Construisons les strings de clefy
    clefy = []
    for ny in range(45):
        a = int(ny / 10)
        b = int(ny % 10)
        clefy.append(str(a) + "_" + str(b))
    # Construisons les strings des 10 capteurs et de leur symétrique
    tableau1 = capteur1 + "_dyn_moda"
    tableau1S = capteur1S + "_dyn_moda"
    tableau2 = capteur2 + "_dyn_moda"
    tableau2S = capteur2S + "_dyn_moda"
    tableau3 = capteur3 + "_dyn_moda"
    tableau3S = capteur3S + "_dyn_moda"
    tableau4 = capteur4 + "_dyn_moda"
    tableau4S = capteur4S + "_dyn_moda"
    tableau5 = capteur5 + "_dyn_moda"
    tableau5S = capteur5S + "_dyn_moda"
    tableau6 = capteur6 + "_dyn_moda"
    tableau6S = capteur6S + "_dyn_moda"
    tableau7 = capteur7 + "_dyn_moda"
    tableau7S = capteur7S + "_dyn_moda"
    tableau8 = capteur8 + "_dyn_moda"
    tableau8S = capteur8S + "_dyn_moda"
    tableau9 = capteur9 + "_dyn_moda"
    tableau9S = capteur9S + "_dyn_moda"
    tableau10 = capteur10 + "_dyn_moda"
    tableau10S = capteur10S + "_dyn_moda"
    # Préparons 10 tableaux à 89 lignes et 4591 colonnes.
    # Les 89 lignes correspondent à 89 valeurs de y de y=-440 à +440
    # avec une valeur tous les 10 cm.
    # Les 4591 colonnes correspondent à une colonne tous les cm en x.
    # Chaque tableau correspond à un capteur (tabk correspond au capteur k, k=1à10)
    lignes, colonnes = 89, 4591
    tab1 = [[0] * colonnes] * lignes
    tab2 = [[0] * colonnes] * lignes
    tab3 = [[0] * colonnes] * lignes
    tab4 = [[0] * colonnes] * lignes
    tab5 = [[0] * colonnes] * lignes
    tab6 = [[0] * colonnes] * lignes
    tab7 = [[0] * colonnes] * lignes
    tab8 = [[0] * colonnes] * lignes
    tab9 = [[0] * colonnes] * lignes
    tab10 = [[0] * colonnes] * lignes

    for ny in range(45):
        # y=0 correspond à l'indice 44 pour les composantes de tabk
        # nny est un indice pour les valeurs de y positives ou nulles. Il varie de 44 à 88.
        # y=(nny-44)*10
        # my est l'indice pour les valeurs de y négatives ou nulles. Il varie de 0 à 44.
        # y=(my-44)*10. (l'indice 44 est vu deux fois, mais tous les deux correspondent à y=0)
        nny = ny + 44
        my = 44 - ny
        # Les tabk (k=1 à 10) sont remplis en faisant appels aux données des
        # deux capteurs symétriques capteurk et capteurkS
        tab1[nny] = data1[capteur1][clefy[ny]][tableau1]
        tab1[my] = data1[capteur1S][clefy[ny]][tableau1S]
        tab2[nny] = data1[capteur2][clefy[ny]][tableau2]
        tab2[my] = data1[capteur2S][clefy[ny]][tableau2S]
        tab3[nny] = data1[capteur3][clefy[ny]][tableau3]
        tab3[my] = data1[capteur3S][clefy[ny]][tableau3S]
        tab4[nny] = data1[capteur4][clefy[ny]][tableau4]
        tab4[my] = data1[capteur4S][clefy[ny]][tableau4S]
        tab5[nny] = data1[capteur5][clefy[ny]][tableau5]
        tab5[my] = data1[capteur5S][clefy[ny]][tableau5S]
        tab6[nny] = data1[capteur6][clefy[ny]][tableau6]
        tab6[my] = data1[capteur6S][clefy[ny]][tableau6S]
        tab7[nny] = data1[capteur7][clefy[ny]][tableau7]
        tab7[my] = data1[capteur7S][clefy[ny]][tableau7S]
        tab8[nny] = data1[capteur8][clefy[ny]][tableau8]
        tab8[my] = data1[capteur8S][clefy[ny]][tableau8S]
        tab9[nny] = data1[capteur9][clefy[ny]][tableau9]
        tab9[my] = data1[capteur9S][clefy[ny]][tableau9S]
        tab10[nny] = data1[capteur10][clefy[ny]][tableau10]
        tab10[my] = data1[capteur10S][clefy[ny]][tableau10S]

    # Il faut maintenant prolonger le tableau suivant les x...

    # Prolongation des fichiers (NI1 -> NI2)
    Tabres = [0] * (NI2 - NI1)
    for ny in range(lignes):
        tab1[ny] = tab1[ny] + Tabres
        tab2[ny] = tab2[ny] + Tabres
        tab3[ny] = tab3[ny] + Tabres
        tab4[ny] = tab4[ny] + Tabres
        tab5[ny] = tab5[ny] + Tabres
        tab6[ny] = tab6[ny] + Tabres
        tab7[ny] = tab7[ny] + Tabres
        tab8[ny] = tab8[ny] + Tabres
        tab9[ny] = tab9[ny] + Tabres
        tab10[ny] = tab10[ny] + Tabres

    # ______________________________________________________________________________
    #       CONSTRUCTION DES FONCTIONS D'INFLUENCE D'UN ESSIEU
    # ______________________________________________________________________________
    #
    data = np.array([tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10], dtype=np.double)
    return torch.from_numpy(data)*1e6


def construct_dataset(V=10, dt=1, batch_size=16):
    dx = int(V * dt)
    data = get_data()
    data = data[:, :, 0:-1:dx]  # [capteur, Y, N]
    data = data.transpose(0, 2)
    dataset_X = []
    dataset_Y = []
    for nyd in range(89):
        for nyg in range(nyd, 89):
            dataset_X.append(data[:, nyg, :] + data[:, nyd, :])
            dataset_Y.append(torch.tensor([(nyg+nyd)/2/89, (nyg-nyd)/89], dtype=torch.double))
    dataset_X = torch.stack(dataset_X)
    dataset_Y = torch.stack(dataset_Y)
    # torch.save(dataset_X, 'dataset_X.pt')
    # torch.save(dataset_Y, 'dataset_Y.pt')

    dataset = TensorDataset(dataset_X, dataset_Y)
    # [B, L, D]
    n_train = len(dataset_X)
    split = n_train // 4
    indices = list(range(n_train))
    random.shuffle(indices)

    # split data to train/validation
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler, drop_last=True)
    return train_loader, valid_loader




