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

# from scipy.signal import savgol_filter
# from scipy import interpolate
plt.close('all')

# Ce script vise à tester un outil d'identification de camions à partir des
# mesures des différents capteurs.

# Pour cela nous tentons de reconstruire la primitive du signal de charge des
# camions qui dans le cas idéal est une fonction en escalier, chaque marche
# correspondant au passage d'un essieu.
#
# Dans ce script, nous commençons par simuler les signaux de mesures suite
# au passage d'un camion.
# Différents outils permettent de modifier ces signaux pour les rendre plus 
# proches de ce qui est effectivement mesuré.
# En particulier il est possible de tronquer le signal et ainsi d'étudier les
# conséquences sur la prévision du signal.

# Ensuite nous commençons la procédure de reconstruction de la primitive du
# signal de charge en commençant par estimer les paramètres importants de la
# trajectoire du camion, la vitesse et l'ordonnée du passage du centre du camion
# sur le tablier du pont.
#
# Enfin,après le calcul des fonctions d'influence essieu sur chaque capteur à
# partir de ces paramètres de trajectoire, et en nous appuyant sur le fait que
# les signaux de mesures sont des convolutions du signal de charge du camion avec
# les fonctions d'influence essieu, nous proposons deux algorythmes basés sur la
# transformée de Fourier discrète (TFD) pour reconstruire la primitive du signal
# de charge des camions.

# Si cette primitive ressemble à une fonction en escalier,l'identification du
# camion, c'est la dire le nombre d'essieux, la position relative des essieux
# et les charges aux essieux peut etre effectuée.


# Il faut donner quelques informations sur le pont
# ____________________________________________________________________________
#           DONNEES PONT 
# _____________________________________________________________________________
# La longueur du pont est L en cm.C'est un entier.
# Pour lepont d'Auzouer c'est 4591cm
# On prolonge le pont de Lmax pour tenir compte de la longueur du camion.
# Dans ce script on retient L+Lmax=8000cm

L = 4591  # cm, longueur du pont
Lmax = 6000 - 4591

NI1 = L
NI2 = L + Lmax

# _____________________________________________________________________________
#           FIN DES DONNEES PONT 
# _____________________________________________________________________________

###############################################################################
#               SIMULATION D'UN PASSAGE DE CAMION
###############################################################################
# La simulation du passage d'un camion comprend deux étapes
# Dans la première étape on construir un signal avec un point par cm
# Dans la deuxième étape on condense le signal pour avoir un point par centiseconde
# Par exemple, avec une vitesse de 10m/s (10cm/cs) On a 8000 points sur le
# premier signal et 800 sur le second)


# _____________________________________________________________________________
#           DONNEES CAMION 
# _____________________________________________________________________________
# Ces données serviront pour simuler le passage d'un camion
# ordonnée du centre du camion en cm. C'est un entier divisible par 10.
y_c = 220
# écartement entre les centres des roues en dm. C'est un entier pair
e_R = 18
# On en déduit l'ordonnée de la roue gauche en cm (par rapport à la gauche du
# tablier). C'est un entier divisible par 10.
y_g = y_c - 10 * int(e_R / 2)
# ordonnée de la roue gauche en dm (par rapport à l'axe du tablier).
# C'est un entier
nyg = int((y_g / 10) + 44)
# ordonnée de la roue droite en dm (par rapport à l'axe du tablier).
# C'est un entier
nyd = nyg + e_R
# vitesse en m/s ou cm/cs (c'est un entier)
V = 10

# Distance en cm, à l'insant 0, entre le début du pont l'essieu numéro e
# (distance avant qu'il ne rentre sur le pont). 
# (nL0e pour e appartenant à {1,2,3,4,5})
nL01 = 0
nL02 = 300
nL03 = 600
nL04 = 750
nL05 = 900

# Charge portée par chaque essieu en tonnes forces
A1d = 5
A2d = 10
A3d = 10
A4d = 10
A5d = 10

# _____________________________________________________________________________
#           FIN DES DONNEES CAMION 
# _____________________________________________________________________________

# La simulation de la réponse des capteurs au passage du camion est construite
# à l'aide des fonctions d'influence sur les différents capteurs,
# c'est à dire de la réponse des capteurs au passage d'un seul essieu sous
# chargement unitaire (1tf)
# Cette fonction d'influence dépend de y_c et e_R.
# Elle comprend NI2=L+Lmax valeurs, c'est à dire une valeur pour chaque cm de
# la position de l'essieu


# Les fonctions d'influence essieu Jc sont la combinaison linéaire
# d'une 1/2 charge unitaire placée en y=nyg et x=int(n*V) (roue gauche, 500kg)
# et d'une 1/2 charge unitaire placée en y=nyd et x=int(n*V) (roue droite, 500kg)
# Il faut donc commencer par importer les fonctions dinfluence monocharge
# unitaire.

# ______________________________________________________________________________
#       IMPORTATION DES FONCTIONS D'INFLUENCE MONOCHARGE UNITAIRE
# ______________________________________________________________________________

#  Le string du chemin d'acces aux fonctions d'influence monocharge unitaire 
# calculées numériquement avec ASTER est
s1 = "./Complet_Isym_ttcapt_V36.json"
# Importons toutes ces fonctions d'influence dans le fichier data1
f = open(s1)
data1 = json.load(f)
f.close()

# f est un dictionnaire.

# On peut demander les clefs de data1 par
# clefs=data1.keys()
# print("Clefs de data1",clefs)
# On obtient
# Clefs de data1 dict_keys(['CO_AXL_67', 'CO_AXL_45', 'CO_AXL_23', 'CO_AXL_89',
# 'CO_T2_P2', 'CO_T2_P4', 'CO_T2_P7', 'CO_T2_P9', 'CO_T3_P4', 'CO_T3_P7'])
# On voit qu'il ya 10 fichiers dans le dictionnaire.
# Ils se distinguent par capteurs
# "Précisons qu'il s'agit des modèles des capteurs dans le modèle numérique du pont
# "Ces fichiers donnent la simulation de la mesure d'un capteur lorsqu'une charge
# unitaire de 1tf est placé en un point (x,y) du tablier
# Le modèle du pont est symétrique par rapport à un axe y=0.
# Les capteurs sont disposés symétriquement.
# Commençons par les renommer en changeant les srings qui les désignent
# Chaque capteur est associé à son symétrique. La notation est redondante
# Par exemple la couple capteur1,capteur1s est identique au couple capteur 4 capteur 4s
# Ceci est fait pour une raison de commodité dans la programation ci dessous.
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

# data1[capteurk] est lui même un dictionnaire dont on peut demander les clefs
# clefs=data1[capteurk].keys()
# print("Clefs de data1 bis",clefs)
# On obtient
# Clefs de data1 bis dict_keys(['0_0', '0_1', '0_2', '0_3', '0_4', '0_5', '0_6',
# '0_7', '0_8', '0_9', '1_0', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', 
# '1_7', '1_8', '1_9', '2_0', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', 
# '2_7', '2_8', '2_9', '3_0', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', 
# '3_7', '3_8', '3_9', '4_0', '4_1', '4_2', '4_3', '4_4'])
# On voit qu'il ya 45 fichiers dans le dictionnaire.
# Ils se distinguent par la valeur de y
# (une valeur de y tous les 10cm de 0 à 440).
# A y en cm on associe tout d'abord l'entier ny=int(y/10)
# Puis à une valeur de ny est associé le string (str(a)+"_"+str(b))
# avec a=int(ny/10) et b=int(ny%10)

# data1[capteurk][clefy] est lui même un dictionnaire dont on peut demander les clefs
# clefs=data1[capteurk][clefy1].keys()
# print("Clefs de data1 ter",clefs)
# On obtient 2 clefs de data1 ter dict_keys(['inst', capteurk+'C_dyn_moda'])
# Le premier tableau 'inst' correspond aux instants de simulation de la
# fonction d'influence avec une vitesse simulée de 36 km/h, toutes les 0,001s.
# Le deuxième tableau est la valeur de la fonction d'influence
# pour le capteur 'capteurk' et la valeur de y correspondant à la clef 'clefy1'.
# La fonction d'influence est fonction de x avec un point par cm
# ( 36km/h=10m/s et la période d'acqusition dans 'inst' est 0,001s).
# Il y a 4591 valeurs pour chaque fichier correspondant à la longueur de 4590cm du pont.

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

# tabc[ny][nx] donne la mesure du capteur c lorsque l'on met une charge unitaire
# sur le tablier du pont au point d'abcisse nx et d'ordonnée ny
# ______________________________________________________________________________
#       FIN DE L'IMPORTATION DES FONCTIONS D'INFLUENCE MONOCHARGE UNITAIRE
# ______________________________________________________________________________

# ______________________________________________________________________________
#       CONSTRUCTION DES FONCTIONS D'INFLUENCE D'UN ESSIEU
# ______________________________________________________________________________
#
#
# Les fonctions d'influence essieu Jc sont la combinaison linéaire
# d'une 1/2 charge unitaire placée en y=nyg et x=int(n*V) (roue gauche, 500kg)
# et d'une 1/2 charge unitaire placée en y=nyd et x=int(n*V) (roue droite, 500kg)
# Le nombre de points de la fonction d'inflence monocharge est égal à NI2=L+Lmax
# Le nombre de points de la fonction d'influence d'un essieu est  N
# Les fonctions d'influence Jc d'un essieu sont des vecteurs à N composantes
# complexes,(pour utiliser les TFD), une composante tous les centisecondes.
# Le passage de NI2 à N nécessite la connaissance de la vitesse V
# et du pas de temps d'enregistrement dt (1cs)
# Calcul de N
dt = 1  # pas de temps en cs
N = round((L + Lmax) / (V * dt))
print('nombre de points du signal=', N)

# En général N<<L+Lmax sauf si Vprime =1 (dt=1)
# Par exemple pour L+Lmax=8000, et Vprime=10cm/cs, N=800.


# Préparons les vecteurs Jc1 àJc10 (capteurs 1 à 10)
# en les initailisant à 0 complexe de double longueur
Jc1 = np.zeros((N), dtype=np.clongdouble)
Jc2 = np.zeros((N), dtype=np.clongdouble)
Jc3 = np.zeros((N), dtype=np.clongdouble)
Jc4 = np.zeros((N), dtype=np.clongdouble)
Jc5 = np.zeros((N), dtype=np.clongdouble)
Jc6 = np.zeros((N), dtype=np.clongdouble)
Jc7 = np.zeros((N), dtype=np.clongdouble)
Jc8 = np.zeros((N), dtype=np.clongdouble)
Jc9 = np.zeros((N), dtype=np.clongdouble)
Jc10 = np.zeros((N), dtype=np.clongdouble)
# Remplissons les composantes des vecteurs Jc.
# tempo est le nx des fonctions d'influence monocharge correspondant à
# l'indice k dans le vecteur Jc.
# Le coefficient 1000 ci dessous correspond à une différence d'unité
# entre les fonctions d'influence monocharge et les fonctions d'influence essieu

for k in range(N):
    tempo = int(k * dt * V)
    Jc1[k] = 1 / 2 * (tab1[nyg][tempo] + tab1[nyd][tempo]) * 1000
    Jc2[k] = 1 / 2 * (tab2[nyg][tempo] + tab2[nyd][tempo]) * 1000
    Jc3[k] = 1 / 2 * (tab3[nyg][tempo] + tab3[nyd][tempo]) * 1000
    Jc4[k] = 1 / 2 * (tab4[nyg][tempo] + tab4[nyd][tempo]) * 1000
    Jc5[k] = 1 / 2 * (tab5[nyg][tempo] + tab5[nyd][tempo]) * 1000
    Jc6[k] = 1 / 2 * (tab6[nyg][tempo] + tab6[nyd][tempo]) * 1000
    Jc7[k] = 1 / 2 * (tab7[nyg][tempo] + tab7[nyd][tempo]) * 1000
    Jc8[k] = 1 / 2 * (tab8[nyg][tempo] + tab8[nyd][tempo]) * 1000
    Jc9[k] = 1 / 2 * (tab9[nyg][tempo] + tab9[nyd][tempo]) * 1000
    Jc10[k] = 1 / 2 * (tab10[nyg][tempo] + tab10[nyd][tempo]) * 1000

plt.plot(Jc1, 'b', lw=2, label=capteur8)
plt.show()

# ______________________________________________________________________________
#       FIN DE LA CONSTRUCTION DES FONCTIONS D'INFLUENCE D'UN ESSIEU
# ______________________________________________________________________________

# 我们先从对一个车轴的模拟值Jc做训练？看能不能给出曲线，找回对应的e_R和yc
# 之后再对整车进行模拟
