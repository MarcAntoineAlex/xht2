import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import json
import timeit
import time
# from scipy.signal import savgol_filter
# from scipy import interpolate
plt.close('all')

#Ce script vise à tester un outil d'identification de camions à partir des 
#mesures des différents capteurs.

#Pour cela nous tentons de reconstruire la primitive du signal de charge des
#camions qui dans le cas idéal est une fonction en escalier, chaque marche 
#correspondant au passage d'un essieu.
#
#Dans ce script, nous commençons par simuler les signaux de mesures suite 
#au passage d'un camion.
# Différents outils permettent de modifier ces signaux pour les rendre plus 
#proches de ce qui est effectivement mesuré.
#En particulier il est possible de tronquer le signal et ainsi d'étudier les 
#conséquences sur la prévision du signal.

#Ensuite nous commençons la procédure de reconstruction de la primitive du 
#signal de charge en commençant par estimer les paramètres importants de la 
#trajectoire du camion, la vitesse et l'ordonnée du passage du centre du camion
#sur le tablier du pont.
#
#Enfin,après le calcul des fonctions d'influence essieu sur chaque capteur à 
#partir de ces paramètres de trajectoire, et en nous appuyant sur le fait que 
#les signaux de mesures sont des convolutions du signal de charge du camion avec
#les fonctions d'influence essieu, nous proposons deux algorythmes basés sur la 
#transformée de Fourier discrète (TFD) pour reconstruire la primitive du signal
#de charge des camions.

#Si cette primitive ressemble à une fonction en escalier,l'identification du 
#camion, c'est la dire le nombre d'essieux, la position relative des essieux 
#et les charges aux essieux peut etre effectuée.




#Il faut donner quelques informations sur le pont
#____________________________________________________________________________
#           DONNEES PONT 
#_____________________________________________________________________________ 
#La longueur du pont est L en cm.C'est un entier.
#Pour lepont d'Auzouer c'est 4591cm 
# On prolonge le pont de Lmax pour tenir compte de la longueur du camion.
#Dans ce script on retient L+Lmax=8000cm
 
L=4591 #cm, longueur du pont
Lmax=6000-4591

NI1=L
NI2=L+Lmax

#_____________________________________________________________________________
#           FIN DES DONNEES PONT 
#_____________________________________________________________________________ 

###############################################################################
#               SIMULATION D'UN PASSAGE DE CAMION
###############################################################################
#La simulation du passage d'un camion comprend deux étapes
#Dans la première étape on construir un signal avec un point par cm
#Dans la deuxième étape on condense le signal pour avoir un point par centiseconde
#Par exemple, avec une vitesse de 10m/s (10cm/cs) On a 8000 points sur le 
#premier signal et 800 sur le second)



#_____________________________________________________________________________
#           DONNEES CAMION 
#_____________________________________________________________________________ 
#Ces données serviront pour simuler le passage d'un camion
#ordonnée du centre du camion en cm. C'est un entier divisible par 10.
y_c=220
#écartement entre les centres des roues en dm. C'est un entier pair
e_R=18 
#On en déduit l'ordonnée de la roue gauche en cm (par rapport à la gauche du 
#tablier). C'est un entier divisible par 10.
y_g=y_c-10*int(e_R/2)
#ordonnée de la roue gauche en dm (par rapport à l'axe du tablier). 
#C'est un entier
nyg=int((y_g/10)+44)
#ordonnée de la roue droite en dm (par rapport à l'axe du tablier). 
#C'est un entier
nyd=nyg+e_R
#vitesse en m/s ou cm/cs (c'est un entier)
V=10

#Distance en cm, à l'insant 0, entre le début du pont l'essieu numéro e  
# (distance avant qu'il ne rentre sur le pont). 
#(nL0e pour e appartenant à {1,2,3,4,5})
nL01=0
nL02=300
nL03=600
nL04=750
nL05=900


#Charge portée par chaque essieu en tonnes forces
A1d=5
A2d=10
A3d=10
A4d=10
A5d=10

#_____________________________________________________________________________
#           FIN DES DONNEES CAMION 
#_____________________________________________________________________________

# La simulation de la réponse des capteurs au passage du camion est construite
#à l'aide des fonctions d'influence sur les différents capteurs, 
#c'est à dire de la réponse des capteurs au passage d'un seul essieu sous 
#chargement unitaire (1tf)
#Cette fonction d'influence dépend de y_c et e_R.
#Elle comprend NI2=L+Lmax valeurs, c'est à dire une valeur pour chaque cm de 
#la position de l'essieu



#Les fonctions d'influence essieu Jc sont la combinaison linéaire  
#d'une 1/2 charge unitaire placée en y=nyg et x=int(n*V) (roue gauche, 500kg)
#et d'une 1/2 charge unitaire placée en y=nyd et x=int(n*V) (roue droite, 500kg)
#Il faut donc commencer par importer les fonctions dinfluence monocharge 
#unitaire.

#______________________________________________________________________________
#       IMPORTATION DES FONCTIONS D'INFLUENCE MONOCHARGE UNITAIRE
#______________________________________________________________________________

#  Le string du chemin d'acces aux fonctions d'influence monocharge unitaire 
#calculées numériquement avec ASTER est 
s1="./Complet_Isym_ttcapt_V36.json"
#Importons toutes ces fonctions d'influence dans le fichier data1
f = open(s1)
data1 = json.load(f)
f.close()

# f est un dictionnaire.

# On peut demander les clefs de data1 par
#clefs=data1.keys()
#print("Clefs de data1",clefs)
#On obtient
#Clefs de data1 dict_keys(['CO_AXL_67', 'CO_AXL_45', 'CO_AXL_23', 'CO_AXL_89',
# 'CO_T2_P2', 'CO_T2_P4', 'CO_T2_P7', 'CO_T2_P9', 'CO_T3_P4', 'CO_T3_P7'])
#On voit qu'il ya 10 fichiers dans le dictionnaire.
#Ils se distinguent par capteurs 
#"Précisons qu'il s'agit des modèles des capteurs dans le modèle numérique du pont
#"Ces fichiers donnent la simulation de la mesure d'un capteur lorsqu'une charge
# unitaire de 1tf est placé en un point (x,y) du tablier
#Le modèle du pont est symétrique par rapport à un axe y=0.
#Les capteurs sont disposés symétriquement.
#Commençons par les renommer en changeant les srings qui les désignent
#Chaque capteur est associé à son symétrique. La notation est redondante
#Par exemple la couple capteur1,capteur1s est identique au couple capteur 4 capteur 4s
#Ceci est fait pour une raison de commodité dans la programation ci dessous.
capteur1= 'CO_T2_P2'
capteur1S='CO_T2_P9'
capteur2= 'CO_T2_P4'
capteur2S='CO_T2_P7'
capteur3= 'CO_T2_P7'
capteur3S='CO_T2_P4'
capteur4= 'CO_T2_P9'
capteur4S='CO_T2_P2'
capteur5= 'CO_T3_P4'
capteur5S='CO_T3_P7'
capteur6= 'CO_T3_P7'
capteur6S='CO_T3_P4'
capteur7= 'CO_AXL_23'
capteur7S='CO_AXL_89'
capteur8= 'CO_AXL_45'
capteur8S='CO_AXL_67'
capteur9= 'CO_AXL_67'
capteur9S='CO_AXL_45'
capteur10= 'CO_AXL_89'
capteur10S='CO_AXL_23'

#data1[capteurk] est lui même un dictionnaire dont on peut demander les clefs
#clefs=data1[capteurk].keys()
#print("Clefs de data1 bis",clefs)
#On obtient
#Clefs de data1 bis dict_keys(['0_0', '0_1', '0_2', '0_3', '0_4', '0_5', '0_6',
# '0_7', '0_8', '0_9', '1_0', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', 
# '1_7', '1_8', '1_9', '2_0', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', 
# '2_7', '2_8', '2_9', '3_0', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', 
# '3_7', '3_8', '3_9', '4_0', '4_1', '4_2', '4_3', '4_4'])
# On voit qu'il ya 45 fichiers dans le dictionnaire.
#Ils se distinguent par la valeur de y 
#(une valeur de y tous les 10cm de 0 à 440). 
#A y en cm on associe tout d'abord l'entier ny=int(y/10)
#Puis à une valeur de ny est associé le string (str(a)+"_"+str(b)) 
#avec a=int(ny/10) et b=int(ny%10)

#data1[capteurk][clefy] est lui même un dictionnaire dont on peut demander les clefs
#clefs=data1[capteurk][clefy1].keys()
#print("Clefs de data1 ter",clefs)
#On obtient 2 clefs de data1 ter dict_keys(['inst', capteurk+'C_dyn_moda'])
#Le premier tableau 'inst' correspond aux instants de simulation de la 
#fonction d'influence avec une vitesse simulée de 36 km/h, toutes les 0,001s.
#Le deuxième tableau est la valeur de la fonction d'influence 
#pour le capteur 'capteurk' et la valeur de y correspondant à la clef 'clefy1'.
# La fonction d'influence est fonction de x avec un point par cm
#( 36km/h=10m/s et la période d'acqusition dans 'inst' est 0,001s).
#Il y a 4591 valeurs pour chaque fichier correspondant à la longueur de 4590cm du pont.

#Construisons les strings de clefy 
clefy=[]
for ny in range(45):
    a=int(ny/10)
    b=int(ny%10)
    clefy.append(str(a)+"_"+str(b))
#Construisons les strings des 10 capteurs et de leur symétrique
tableau1=capteur1+"_dyn_moda"
tableau1S=capteur1S+"_dyn_moda"
tableau2=capteur2+"_dyn_moda"
tableau2S=capteur2S+"_dyn_moda"
tableau3=capteur3+"_dyn_moda"
tableau3S=capteur3S+"_dyn_moda"
tableau4=capteur4+"_dyn_moda"
tableau4S=capteur4S+"_dyn_moda"
tableau5=capteur5+"_dyn_moda"
tableau5S=capteur5S+"_dyn_moda"
tableau6=capteur6+"_dyn_moda"
tableau6S=capteur6S+"_dyn_moda"
tableau7=capteur7+"_dyn_moda"
tableau7S=capteur7S+"_dyn_moda"
tableau8=capteur8+"_dyn_moda"
tableau8S=capteur8S+"_dyn_moda"
tableau9=capteur9+"_dyn_moda"
tableau9S=capteur9S+"_dyn_moda"
tableau10=capteur10+"_dyn_moda"
tableau10S=capteur10S+"_dyn_moda"
#Préparons 10 tableaux à 89 lignes et 4591 colonnes.
#Les 89 lignes correspondent à 89 valeurs de y de y=-440 à +440 
#avec une valeur tous les 10 cm.
#Les 4591 colonnes correspondent à une colonne tous les cm en x.
#Chaque tableau correspond à un capteur (tabk correspond au capteur k, k=1à10) 
lignes, colonnes=89, 4591
tab1=[[0]*colonnes]*lignes
tab2=[[0]*colonnes]*lignes
tab3=[[0]*colonnes]*lignes
tab4=[[0]*colonnes]*lignes
tab5=[[0]*colonnes]*lignes
tab6=[[0]*colonnes]*lignes
tab7=[[0]*colonnes]*lignes
tab8=[[0]*colonnes]*lignes
tab9=[[0]*colonnes]*lignes
tab10=[[0]*colonnes]*lignes


for ny in range(45):
#y=0 correspond à l'indice 44 pour les composantes de tabk
#nny est un indice pour les valeurs de y positives ou nulles. Il varie de 44 à 88.
#y=(nny-44)*10
#my est l'indice pour les valeurs de y négatives ou nulles. Il varie de 0 à 44.
#y=(my-44)*10. (l'indice 44 est vu deux fois, mais tous les deux correspondent à y=0)
    nny=ny+44
    my=44-ny
# Les tabk (k=1 à 10) sont remplis en faisant appels aux données des  
#deux capteurs symétriques capteurk et capteurkS   
    tab1[nny]=data1[capteur1][clefy[ny]][tableau1]
    tab1[my]=data1[capteur1S][clefy[ny]][tableau1S]
    tab2[nny]=data1[capteur2][clefy[ny]][tableau2]
    tab2[my]=data1[capteur2S][clefy[ny]][tableau2S]
    tab3[nny]=data1[capteur3][clefy[ny]][tableau3]
    tab3[my]=data1[capteur3S][clefy[ny]][tableau3S]
    tab4[nny]=data1[capteur4][clefy[ny]][tableau4]
    tab4[my]=data1[capteur4S][clefy[ny]][tableau4S]
    tab5[nny]=data1[capteur5][clefy[ny]][tableau5]
    tab5[my]=data1[capteur5S][clefy[ny]][tableau5S]
    tab6[nny]=data1[capteur6][clefy[ny]][tableau6]
    tab6[my]=data1[capteur6S][clefy[ny]][tableau6S]
    tab7[nny]=data1[capteur7][clefy[ny]][tableau7]
    tab7[my]=data1[capteur7S][clefy[ny]][tableau7S]
    tab8[nny]=data1[capteur8][clefy[ny]][tableau8]
    tab8[my]=data1[capteur8S][clefy[ny]][tableau8S]
    tab9[nny]=data1[capteur9][clefy[ny]][tableau9]
    tab9[my]=data1[capteur9S][clefy[ny]][tableau9S]
    tab10[nny]=data1[capteur10][clefy[ny]][tableau10]
    tab10[my]=data1[capteur10S][clefy[ny]][tableau10S]

#Il faut maintenant prolonger le tableau suivant les x...

#Prolongation des fichiers (NI1 -> NI2)
Tabres=[0]*(NI2-NI1)
for ny in range(lignes):
    tab1[ny]=tab1[ny]+Tabres
    tab2[ny]=tab2[ny]+Tabres
    tab3[ny]=tab3[ny]+Tabres
    tab4[ny]=tab4[ny]+Tabres
    tab5[ny]=tab5[ny]+Tabres
    tab6[ny]=tab6[ny]+Tabres
    tab7[ny]=tab7[ny]+Tabres
    tab8[ny]=tab8[ny]+Tabres
    tab9[ny]=tab9[ny]+Tabres
    tab10[ny]=tab10[ny]+Tabres

#tabc[ny][nx] donne la mesure du capteur c lorsque l'on met une charge unitaire
# sur le tablier du pont au point d'abcisse nx et d'ordonnée ny
#______________________________________________________________________________
#       FIN DE L'IMPORTATION DES FONCTIONS D'INFLUENCE MONOCHARGE UNITAIRE
#______________________________________________________________________________

#______________________________________________________________________________
#       CONSTRUCTION DES FONCTIONS D'INFLUENCE D'UN ESSIEU
#______________________________________________________________________________
#
#
#Les fonctions d'influence essieu Jc sont la combinaison linéaire  
#d'une 1/2 charge unitaire placée en y=nyg et x=int(n*V) (roue gauche, 500kg) 
#et d'une 1/2 charge unitaire placée en y=nyd et x=int(n*V) (roue droite, 500kg)
#Le nombre de points de la fonction d'inflence monocharge est égal à NI2=L+Lmax
#Le nombre de points de la fonction d'influence d'un essieu est  N
#Les fonctions d'influence Jc d'un essieu sont des vecteurs à N composantes 
#complexes,(pour utiliser les TFD), une composante tous les centisecondes.
#Le passage de NI2 à N nécessite la connaissance de la vitesse V 
#et du pas de temps d'enregistrement dt (1cs)
#Calcul de N
dt=1 #pas de temps en cs
N=round((L+Lmax)/(V*dt))
print('nombre de points du signal=',N)

#En général N<<L+Lmax sauf si Vprime =1 (dt=1)
#Par exemple pour L+Lmax=8000, et Vprime=10cm/cs, N=800.




# Préparons les vecteurs Jc1 àJc10 (capteurs 1 à 10)
#en les initailisant à 0 complexe de double longueur
Jc1=np.zeros((N), dtype = np.clongdouble)
Jc2=np.zeros((N), dtype = np.clongdouble)
Jc3=np.zeros((N), dtype = np.clongdouble)
Jc4=np.zeros((N), dtype = np.clongdouble)
Jc5=np.zeros((N), dtype = np.clongdouble)
Jc6=np.zeros((N), dtype = np.clongdouble)
Jc7=np.zeros((N), dtype = np.clongdouble)
Jc8=np.zeros((N), dtype = np.clongdouble)
Jc9=np.zeros((N), dtype = np.clongdouble)
Jc10=np.zeros((N), dtype = np.clongdouble)
#Remplissons les composantes des vecteurs Jc.
#tempo est le nx des fonctions d'influence monocharge correspondant à 
#l'indice k dans le vecteur Jc.
#Le coefficient 1000 ci dessous correspond à une différence d'unité 
#entre les fonctions d'influence monocharge et les fonctions d'influence essieu

for k in range(N):        
    tempo=int(k*dt*V)
    Jc1[k]=1/2*(tab1[nyg][tempo]+tab1[nyd][tempo])*1000
    Jc2[k]=1/2*(tab2[nyg][tempo]+tab2[nyd][tempo])*1000
    Jc3[k]=1/2*(tab3[nyg][tempo]+tab3[nyd][tempo])*1000
    Jc4[k]=1/2*(tab4[nyg][tempo]+tab4[nyd][tempo])*1000
    Jc5[k]=1/2*(tab5[nyg][tempo]+tab5[nyd][tempo])*1000
    Jc6[k]=1/2*(tab6[nyg][tempo]+tab6[nyd][tempo])*1000
    Jc7[k]=1/2*(tab7[nyg][tempo]+tab7[nyd][tempo])*1000
    Jc8[k]=1/2*(tab8[nyg][tempo]+tab8[nyd][tempo])*1000
    Jc9[k]=1/2*(tab9[nyg][tempo]+tab9[nyd][tempo])*1000
    Jc10[k]=1/2*(tab10[nyg][tempo]+tab10[nyd][tempo])*1000
# ATTENTION! modifions le dernier terme pour éviter une première composante 0
#dans le vecteur "Transformée de Fourier discrète" de la "Dérivée discrète" de Jc
#En effet dJc1tilde[0]=Jc1[N-1]-Jc1[0]=0 sans la modification.
#Ceci entraine une division par 0 lors de l'opération de déconvolution avec la 
#deuxième méthode ci dessous
# Il faudra étudier les conséquences du choix de cette modification
Jc1[N-1]=Jc1[N-1]+0.000000001
Jc2[N-1]=Jc2[N-1]+0.000000001
Jc3[N-1]=Jc3[N-1]+0.000000001
Jc4[N-1]=Jc4[N-1]+0.000000001
Jc5[N-1]=Jc5[N-1]+0.000000001
Jc6[N-1]=Jc6[N-1]+0.000000001
Jc7[N-1]=Jc7[N-1]+0.000000001
Jc8[N-1]=Jc8[N-1]+0.000000001
Jc9[N-1]=Jc9[N-1]+0.000000001
Jc10[N-1]=Jc10[N-1]+0.000000001




#______________________________________________________________________________
#       FIN DE LA CONSTRUCTION DES FONCTIONS D'INFLUENCE D'UN ESSIEU
#______________________________________________________________________________


      

#Comme le problème que nous devons résoudre est celui de l'identification d'un camion
# à partir des mesures des différents capteurs positionnés sur le pont, 
#il nous faut maintenant entrer ici ces mesures.
#Avant d'entrer des mesures réelles, nous allons utiliser des simulations de mesures
#obtenues en simulant le passage d'un camion.
#Cette simulation se fait en considérant une combinaison linéaire de 
#différentes fonctions d'influence essieu, la charge à l'essieu étant 
#le coefficient multiplicatif associé à la fonction d'influence de cet essieu.
#Une petite complication intervient pour tenir compte du fait que les essieux 
#entrent et sortent progressivement du pont.
#Cette combinaison linéaire donne des vecteurs (tabI31 à tabI40) à NI2=L+Lmax composantes 
#Il sera donc nécessaire de condenser ces vecteurs en des vecteurs à N composantes
#pour simuler des mesures réelles

# Pour avoir des simulations de mesures proche des mesures réelles, 
#il est utile de pouvoir tronquer et/ou shifter les simulations initiales.


#_____________________________________________________________________________
#            DEFINITION DES FONCTIONS TRONCATURES ET  SHIFTS
#_____________________________________________________________________________


# paramètres de troncature et shift
#cut1 est le nombre de points de la courbe L, au début, ramenés à zéro
#cut2 est le nombre de points de la courbe L, à la fin, ramenés à zéro
#nshift est le décalage vers la droite de la courbe L qui devient la courbe e
cut1=0
cut2=0
nshift=0

def troncature(L,cut1,cut2):
    for k in range(cut1):
        L[k]=0
    for k in range(len(L)-cut2,len(L)):
        L[k]=0
    return L
  
def shift(L,n):
    e = np.zeros_like(L)
    if n==0: return L
    if n>0:
        e[:n] = np.zeros(n)
        e[n:] = L[:-n]
    else:
        e[n:] = np.zeros(-n)
        e[:n] = L[-n:]
    return e

#_____________________________________________________________________________
#            FIN DES DEFINITIONS DES FONCTIONS TRONCATURE ET  SHIFT
#_____________________________________________________________________________


#______________________________________________________________________________
#       SIMULATION DES MESURES DES CAPTEURS LORS DU PASSAGE D'UN CAMION
#______________________________________________________________________________


 #Nous commençons par initier à 0 les vecteurs tabI31 àtabI40

t=np.zeros(NI2)
tabI31=np.zeros((NI2))
tabI32=np.zeros((NI2))
tabI33=np.zeros((NI2))
tabI34=np.zeros((NI2))
tabI35=np.zeros((NI2))
tabI36=np.zeros((NI2))
tabI37=np.zeros((NI2))
tabI38=np.zeros((NI2))
tabI39=np.zeros((NI2))
tabI40=np.zeros((NI2))

#Pour i entre nL0 et nL02-1 il n'ya qu'un seul essieu sur le pont, l'essieu avant 
#Les simulations de mesures sur les capteurs sont donc données ci dessous.
#Les instants correspondants sont donnés dans le vecteur t[i]  
    
for i in range(nL01,nL02):
    t[i] = i/V    
    tabI31[i]=((A1d/2)*tab1[nyg][i-nL01]+(A1d/2)*tab1[nyd][i-nL01])*1000
    tabI32[i]=((A1d/2)*tab2[nyg][i-nL01]+(A1d/2)*tab2[nyd][i-nL01])*1000
    tabI33[i]=((A1d/2)*tab3[nyg][i-nL01]+(A1d/2)*tab3[nyd][i-nL01])*1000
    tabI34[i]=((A1d/2)*tab4[nyg][i-nL01]+(A1d/2)*tab4[nyd][i-nL01])*1000
    tabI35[i]=((A1d/2)*tab5[nyg][i-nL01]+(A1d/2)*tab5[nyd][i-nL01])*1000
    tabI36[i]=((A1d/2)*tab6[nyg][i-nL01]+(A1d/2)*tab6[nyd][i-nL01])*1000
    tabI37[i]=((A1d/2)*tab7[nyg][i-nL01]+(A1d/2)*tab7[nyd][i-nL01])*1000
    tabI38[i]=((A1d/2)*tab8[nyg][i-nL01]+(A1d/2)*tab8[nyd][i-nL01])*1000
    tabI39[i]=((A1d/2)*tab9[nyg][i-nL01]+(A1d/2)*tab9[nyd][i-nL01])*1000
    tabI40[i]=((A1d/2)*tab10[nyg][i-nL01]+(A1d/2)*tab10[nyd][i-nL01])*1000
 
#Nous avons ensuite deux essieux pour i entre nL02 et nL03-1   

for i in range(nL02,nL03):
     t[i] = i/V    
     tabI31[i]=((A1d/2)*tab1[nyg][i-nL01]+(A1d/2)*tab1[nyd][i-nL01]+(A2d/2)*tab1[nyg][i-nL02]+(A2d/2)*tab1[nyd][i-nL02])*1000
     tabI32[i]=((A1d/2)*tab2[nyg][i-nL01]+(A1d/2)*tab2[nyd][i-nL01]+(A2d/2)*tab2[nyg][i-nL02]+(A2d/2)*tab2[nyd][i-nL02])*1000
     tabI33[i]=((A1d/2)*tab3[nyg][i-nL01]+(A1d/2)*tab3[nyd][i-nL01]+(A2d/2)*tab3[nyg][i-nL02]+(A2d/2)*tab3[nyd][i-nL02])*1000
     tabI34[i]=((A1d/2)*tab4[nyg][i-nL01]+(A1d/2)*tab4[nyd][i-nL01]+(A2d/2)*tab4[nyg][i-nL02]+(A2d/2)*tab4[nyd][i-nL02])*1000
     tabI35[i]=((A1d/2)*tab5[nyg][i-nL01]+(A1d/2)*tab5[nyd][i-nL01]+(A2d/2)*tab5[nyg][i-nL02]+(A2d/2)*tab5[nyd][i-nL02])*1000
     tabI36[i]=((A1d/2)*tab6[nyg][i-nL01]+(A1d/2)*tab6[nyd][i-nL01]+(A2d/2)*tab6[nyg][i-nL02]+(A2d/2)*tab6[nyd][i-nL02])*1000
     tabI37[i]=((A1d/2)*tab7[nyg][i-nL01]+(A1d/2)*tab7[nyd][i-nL01]+(A2d/2)*tab7[nyg][i-nL02]+(A2d/2)*tab7[nyd][i-nL02])*1000
     tabI38[i]=((A1d/2)*tab8[nyg][i-nL01]+(A1d/2)*tab8[nyd][i-nL01]+(A2d/2)*tab8[nyg][i-nL02]+(A2d/2)*tab8[nyd][i-nL02])*1000
     tabI39[i]=((A1d/2)*tab9[nyg][i-nL01]+(A1d/2)*tab9[nyd][i-nL01]+(A2d/2)*tab9[nyg][i-nL02]+(A2d/2)*tab9[nyd][i-nL02])*1000
     tabI40[i]=((A1d/2)*tab10[nyg][i-nL01]+(A1d/2)*tab10[nyd][i-nL01]+(A2d/2)*tab10[nyg][i-nL02]+(A2d/2)*tab10[nyd][i-nL02])*1000

# #Il ya 3 essieux pour i entre nL03 et nL04-1 

for i in range(nL03,nL04):
    t[i] = i/V    
    tabI31[i]=((A1d/2)*tab1[nyg][i-nL01]+(A1d/2)*tab1[nyd][i-nL01]+(A2d/2)*tab1[nyg][i-nL02]+(A2d/2)*tab1[nyd][i-nL02])*1000+((A3d/2)*tab1[nyg][i-nL03]+(A3d/2)*tab1[nyd][i-nL03])*1000
    tabI32[i]=((A1d/2)*tab2[nyg][i-nL01]+(A1d/2)*tab2[nyd][i-nL01]+(A2d/2)*tab2[nyg][i-nL02]+(A2d/2)*tab2[nyd][i-nL02])*1000+((A3d/2)*tab2[nyg][i-nL03]+(A3d/2)*tab2[nyd][i-nL03])*1000
    tabI33[i]=((A1d/2)*tab3[nyg][i-nL01]+(A1d/2)*tab3[nyd][i-nL01]+(A2d/2)*tab3[nyg][i-nL02]+(A2d/2)*tab3[nyd][i-nL02])*1000+((A3d/2)*tab3[nyg][i-nL03]+(A3d/2)*tab3[nyd][i-nL03])*1000
    tabI34[i]=((A1d/2)*tab4[nyg][i-nL01]+(A1d/2)*tab4[nyd][i-nL01]+(A2d/2)*tab4[nyg][i-nL02]+(A2d/2)*tab4[nyd][i-nL02])*1000+((A3d/2)*tab4[nyg][i-nL03]+(A3d/2)*tab4[nyd][i-nL03])*1000
    tabI35[i]=((A1d/2)*tab5[nyg][i-nL01]+(A1d/2)*tab5[nyd][i-nL01]+(A2d/2)*tab5[nyg][i-nL02]+(A2d/2)*tab5[nyd][i-nL02])*1000+((A3d/2)*tab5[nyg][i-nL03]+(A3d/2)*tab5[nyd][i-nL03])*1000
    tabI36[i]=((A1d/2)*tab6[nyg][i-nL01]+(A1d/2)*tab6[nyd][i-nL01]+(A2d/2)*tab6[nyg][i-nL02]+(A2d/2)*tab6[nyd][i-nL02])*1000+((A3d/2)*tab6[nyg][i-nL03]+(A3d/2)*tab6[nyd][i-nL03])*1000
    tabI37[i]=((A1d/2)*tab7[nyg][i-nL01]+(A1d/2)*tab7[nyd][i-nL01]+(A2d/2)*tab7[nyg][i-nL02]+(A2d/2)*tab7[nyd][i-nL02])*1000+((A3d/2)*tab7[nyg][i-nL03]+(A3d/2)*tab7[nyd][i-nL03])*1000
    tabI38[i]=((A1d/2)*tab8[nyg][i-nL01]+(A1d/2)*tab8[nyd][i-nL01]+(A2d/2)*tab8[nyg][i-nL02]+(A2d/2)*tab8[nyd][i-nL02])*1000+((A3d/2)*tab8[nyg][i-nL03]+(A3d/2)*tab8[nyd][i-nL03])*1000
    tabI39[i]=((A1d/2)*tab9[nyg][i-nL01]+(A1d/2)*tab9[nyd][i-nL01]+(A2d/2)*tab9[nyg][i-nL02]+(A2d/2)*tab9[nyd][i-nL02])*1000+((A3d/2)*tab9[nyg][i-nL03]+(A3d/2)*tab9[nyd][i-nL03])*1000
    tabI40[i]=((A1d/2)*tab10[nyg][i-nL01]+(A1d/2)*tab10[nyd][i-nL01]+(A2d/2)*tab10[nyg][i-nL02]+(A2d/2)*tab10[nyd][i-nL02])*1000+((A3d/2)*tab10[nyg][i-nL03]+(A3d/2)*tab10[nyd][i-nL03])*1000

# #Puis 4 essieux pour i entre nL04 et nL05-1  

for i in range(nL04,nL05):
    t[i] = i/V    
    tabI31[i]=((A1d/2)*tab1[nyg][i-nL01]+(A1d/2)*tab1[nyd][i-nL01]+(A2d/2)*tab1[nyg][i-nL02]+(A2d/2)*tab1[nyd][i-nL02])*1000+((A3d/2)*tab1[nyg][i-nL03]+(A3d/2)*tab1[nyd][i-nL03]+(A4d/2)*tab1[nyg][i-nL04]+(A4d/2)*tab1[nyd][i-nL04])*1000
    tabI32[i]=((A1d/2)*tab2[nyg][i-nL01]+(A1d/2)*tab2[nyd][i-nL01]+(A2d/2)*tab2[nyg][i-nL02]+(A2d/2)*tab2[nyd][i-nL02])*1000+((A3d/2)*tab2[nyg][i-nL03]+(A3d/2)*tab2[nyd][i-nL03]+(A4d/2)*tab2[nyg][i-nL04]+(A4d/2)*tab2[nyd][i-nL04])*1000
    tabI33[i]=((A1d/2)*tab3[nyg][i-nL01]+(A1d/2)*tab3[nyd][i-nL01]+(A2d/2)*tab3[nyg][i-nL02]+(A2d/2)*tab3[nyd][i-nL02])*1000+((A3d/2)*tab3[nyg][i-nL03]+(A3d/2)*tab3[nyd][i-nL03]+(A4d/2)*tab3[nyg][i-nL04]+(A4d/2)*tab3[nyd][i-nL04])*1000
    tabI34[i]=((A1d/2)*tab4[nyg][i-nL01]+(A1d/2)*tab4[nyd][i-nL01]+(A2d/2)*tab4[nyg][i-nL02]+(A2d/2)*tab4[nyd][i-nL02])*1000+((A3d/2)*tab4[nyg][i-nL03]+(A3d/2)*tab4[nyd][i-nL03]+(A4d/2)*tab4[nyg][i-nL04]+(A4d/2)*tab4[nyd][i-nL04])*1000
    tabI35[i]=((A1d/2)*tab5[nyg][i-nL01]+(A1d/2)*tab5[nyd][i-nL01]+(A2d/2)*tab5[nyg][i-nL02]+(A2d/2)*tab5[nyd][i-nL02])*1000+((A3d/2)*tab5[nyg][i-nL03]+(A3d/2)*tab5[nyd][i-nL03]+(A4d/2)*tab5[nyg][i-nL04]+(A4d/2)*tab5[nyd][i-nL04])*1000
    tabI36[i]=((A1d/2)*tab6[nyg][i-nL01]+(A1d/2)*tab6[nyd][i-nL01]+(A2d/2)*tab6[nyg][i-nL02]+(A2d/2)*tab6[nyd][i-nL02])*1000+((A3d/2)*tab6[nyg][i-nL03]+(A3d/2)*tab6[nyd][i-nL03]+(A4d/2)*tab6[nyg][i-nL04]+(A4d/2)*tab6[nyd][i-nL04])*1000
    tabI37[i]=((A1d/2)*tab7[nyg][i-nL01]+(A1d/2)*tab7[nyd][i-nL01]+(A2d/2)*tab7[nyg][i-nL02]+(A2d/2)*tab7[nyd][i-nL02])*1000+((A3d/2)*tab7[nyg][i-nL03]+(A3d/2)*tab7[nyd][i-nL03]+(A4d/2)*tab7[nyg][i-nL04]+(A4d/2)*tab7[nyd][i-nL04])*1000
    tabI38[i]=((A1d/2)*tab8[nyg][i-nL01]+(A1d/2)*tab8[nyd][i-nL01]+(A2d/2)*tab8[nyg][i-nL02]+(A2d/2)*tab8[nyd][i-nL02])*1000+((A3d/2)*tab8[nyg][i-nL03]+(A3d/2)*tab8[nyd][i-nL03]+(A4d/2)*tab8[nyg][i-nL04]+(A4d/2)*tab8[nyd][i-nL04])*1000
    tabI39[i]=((A1d/2)*tab9[nyg][i-nL01]+(A1d/2)*tab9[nyd][i-nL01]+(A2d/2)*tab9[nyg][i-nL02]+(A2d/2)*tab9[nyd][i-nL02])*1000+((A3d/2)*tab9[nyg][i-nL03]+(A3d/2)*tab9[nyd][i-nL03]+(A4d/2)*tab9[nyg][i-nL04]+(A4d/2)*tab9[nyd][i-nL04])*1000
    tabI40[i]=((A1d/2)*tab10[nyg][i-nL01]+(A1d/2)*tab10[nyd][i-nL01]+(A2d/2)*tab10[nyg][i-nL02]+(A2d/2)*tab10[nyd][i-nL02])*1000+((A3d/2)*tab10[nyg][i-nL03]+(A3d/2)*tab10[nyd][i-nL03]+(A4d/2)*tab10[nyg][i-nL04]+(A4d/2)*tab10[nyd][i-nL04])*1000

#Puis 5 essieux pour les autres valeurs de i.
#(Les différents vecteurs d'influence monocharge ont été prolongés par des zéros jusqu'à NI2=L+Lmax)

for i in range(nL05,NI2):
    t[i] = i/V    
    tabI31[i]=((A1d/2)*tab1[nyg][i-nL01]+(A1d/2)*tab1[nyd][i-nL01]+(A2d/2)*tab1[nyg][i-nL02]+(A2d/2)*tab1[nyd][i-nL02])*1000+((A3d/2)*tab1[nyg][i-nL03]+(A3d/2)*tab1[nyd][i-nL03]+(A4d/2)*tab1[nyg][i-nL04]+(A4d/2)*tab1[nyd][i-nL04])*1000+((A5d/2)*tab1[nyg][i-nL05]+(A5d/2)*tab1[nyd][i-nL05])*1000
    tabI32[i]=((A1d/2)*tab2[nyg][i-nL01]+(A1d/2)*tab2[nyd][i-nL01]+(A2d/2)*tab2[nyg][i-nL02]+(A2d/2)*tab2[nyd][i-nL02])*1000+((A3d/2)*tab2[nyg][i-nL03]+(A3d/2)*tab2[nyd][i-nL03]+(A4d/2)*tab2[nyg][i-nL04]+(A4d/2)*tab2[nyd][i-nL04])*1000+((A5d/2)*tab2[nyg][i-nL05]+(A5d/2)*tab2[nyd][i-nL05])*1000    
    tabI33[i]=((A1d/2)*tab3[nyg][i-nL01]+(A1d/2)*tab3[nyd][i-nL01]+(A2d/2)*tab3[nyg][i-nL02]+(A2d/2)*tab3[nyd][i-nL02])*1000+((A3d/2)*tab3[nyg][i-nL03]+(A3d/2)*tab3[nyd][i-nL03]+(A4d/2)*tab3[nyg][i-nL04]+(A4d/2)*tab3[nyd][i-nL04])*1000+((A5d/2)*tab3[nyg][i-nL05]+(A5d/2)*tab3[nyd][i-nL05])*1000    
    tabI34[i]=((A1d/2)*tab4[nyg][i-nL01]+(A1d/2)*tab4[nyd][i-nL01]+(A2d/2)*tab4[nyg][i-nL02]+(A2d/2)*tab4[nyd][i-nL02])*1000+((A3d/2)*tab4[nyg][i-nL03]+(A3d/2)*tab4[nyd][i-nL03]+(A4d/2)*tab4[nyg][i-nL04]+(A4d/2)*tab4[nyd][i-nL04])*1000+((A5d/2)*tab4[nyg][i-nL05]+(A5d/2)*tab4[nyd][i-nL05])*1000    
    tabI35[i]=((A1d/2)*tab5[nyg][i-nL01]+(A1d/2)*tab5[nyd][i-nL01]+(A2d/2)*tab5[nyg][i-nL02]+(A2d/2)*tab5[nyd][i-nL02])*1000+((A3d/2)*tab5[nyg][i-nL03]+(A3d/2)*tab5[nyd][i-nL03]+(A4d/2)*tab5[nyg][i-nL04]+(A4d/2)*tab5[nyd][i-nL04])*1000+((A5d/2)*tab5[nyg][i-nL05]+(A5d/2)*tab5[nyd][i-nL05])*1000    
    tabI36[i]=((A1d/2)*tab6[nyg][i-nL01]+(A1d/2)*tab6[nyd][i-nL01]+(A2d/2)*tab6[nyg][i-nL02]+(A2d/2)*tab6[nyd][i-nL02])*1000+((A3d/2)*tab6[nyg][i-nL03]+(A3d/2)*tab6[nyd][i-nL03]+(A4d/2)*tab6[nyg][i-nL04]+(A4d/2)*tab6[nyd][i-nL04])*1000+((A5d/2)*tab6[nyg][i-nL05]+(A5d/2)*tab6[nyd][i-nL05])*1000    
    tabI37[i]=((A1d/2)*tab7[nyg][i-nL01]+(A1d/2)*tab7[nyd][i-nL01]+(A2d/2)*tab7[nyg][i-nL02]+(A2d/2)*tab7[nyd][i-nL02])*1000+((A3d/2)*tab7[nyg][i-nL03]+(A3d/2)*tab7[nyd][i-nL03]+(A4d/2)*tab7[nyg][i-nL04]+(A4d/2)*tab7[nyd][i-nL04])*1000+((A5d/2)*tab7[nyg][i-nL05]+(A5d/2)*tab7[nyd][i-nL05])*1000    
    tabI38[i]=((A1d/2)*tab8[nyg][i-nL01]+(A1d/2)*tab8[nyd][i-nL01]+(A2d/2)*tab8[nyg][i-nL02]+(A2d/2)*tab8[nyd][i-nL02])*1000+((A3d/2)*tab8[nyg][i-nL03]+(A3d/2)*tab8[nyd][i-nL03]+(A4d/2)*tab8[nyg][i-nL04]+(A4d/2)*tab8[nyd][i-nL04])*1000+((A5d/2)*tab8[nyg][i-nL05]+(A5d/2)*tab8[nyd][i-nL05])*1000    
    tabI39[i]=((A1d/2)*tab9[nyg][i-nL01]+(A1d/2)*tab9[nyd][i-nL01]+(A2d/2)*tab9[nyg][i-nL02]+(A2d/2)*tab9[nyd][i-nL02])*1000+((A3d/2)*tab9[nyg][i-nL03]+(A3d/2)*tab9[nyd][i-nL03]+(A4d/2)*tab9[nyg][i-nL04]+(A4d/2)*tab9[nyd][i-nL04])*1000+((A5d/2)*tab9[nyg][i-nL05]+(A5d/2)*tab9[nyd][i-nL05])*1000    
    tabI40[i]=((A1d/2)*tab10[nyg][i-nL01]+(A1d/2)*tab10[nyd][i-nL01]+(A2d/2)*tab10[nyg][i-nL02]+(A2d/2)*tab10[nyd][i-nL02])*1000+((A3d/2)*tab10[nyg][i-nL03]+(A3d/2)*tab10[nyd][i-nL03]+(A4d/2)*tab10[nyg][i-nL04]+(A4d/2)*tab10[nyd][i-nL04])*1000+((A5d/2)*tab10[nyg][i-nL05]+(A5d/2)*tab10[nyd][i-nL05])*1000

#______________________________________________________________________________
#    FIN DE LA SIMULATION DES MESURES DES CAPTEURS LORS DU PASSAGE D'UN CAMION
#______________________________________________________________________________


#______________________________________________________________________________
#CONDENSATION DES SIMULATIONS DES MESURES DES CAPTEURS EN VECTEURS A N COMPOSANTES
#_TRONCATURE ET SHIFT
#_____________________________________________________________________________

#Ces  nouveaux vecteurs simulant les mesures à N composantes seront notés 
# Muc1 à Muc10.
#La kième composante de Muc.. sera la composante numéro int(k*dt*V) de TabI..


# Initialisons à 0 les vecteur Muc avec des complexes de double longueur

Muc1=np.zeros((N), dtype = np.clongdouble)
Muc2=np.zeros((N), dtype = np.clongdouble)
Muc3=np.zeros((N), dtype = np.clongdouble)
Muc4=np.zeros((N), dtype = np.clongdouble)
Muc5=np.zeros((N), dtype = np.clongdouble)
Muc6=np.zeros((N), dtype = np.clongdouble)
Muc7=np.zeros((N), dtype = np.clongdouble)
Muc8=np.zeros((N), dtype = np.clongdouble)
Muc9=np.zeros((N), dtype = np.clongdouble)
Muc10=np.zeros((N), dtype = np.clongdouble)
# Remplissons ces vecteurs avec les composantes associées de la simulation d'origine 
#sans interpolation) 
for k in range(N):
    Muc1[k]=tabI31[int(k*dt*V)]
    Muc2[k]=tabI32[int(k*dt*V)]
    Muc3[k]=tabI33[int(k*dt*V)]
    Muc4[k]=tabI34[int(k*dt*V)]
    Muc5[k]=tabI35[int(k*dt*V)]
    Muc6[k]=tabI36[int(k*dt*V)]
    Muc7[k]=tabI37[int(k*dt*V)]
    Muc8[k]=tabI38[int(k*dt*V)]
    Muc9[k]=tabI39[int(k*dt*V)]
    Muc10[k]=tabI40[int(k*dt*V)]

#Le signal peut éventuellement subir une troncature pour ressembler d'avantage à un vrai signal de mesures
#(Parametres de troncature, cut1 et cut2, donnés au début du script) 
Muc1=troncature(Muc1,cut1,cut2)
Muc2=troncature(Muc2,cut1,cut2)
Muc3=troncature(Muc3,cut1,cut2)
Muc4=troncature(Muc4,cut1,cut2)
Muc5=troncature(Muc5,cut1,cut2)
Muc6=troncature(Muc6,cut1,cut2)
Muc7=troncature(Muc7,cut1,cut2)
Muc8=troncature(Muc8,cut1,cut2)
Muc9=troncature(Muc9,cut1,cut2)
Muc10=troncature(Muc10,cut1,cut2)
# # Il peut etre aussi shifter
# #(Paramètre de shift, nshift, donné au début du script) 
Muc1=shift(Muc1, nshift)
Muc2=shift(Muc2, nshift)
Muc3=shift(Muc3, nshift)
Muc4=shift(Muc4, nshift)
Muc5=shift(Muc5, nshift)
Muc6=shift(Muc6, nshift)
Muc7=shift(Muc7, nshift)
Muc8=shift(Muc8, nshift)
Muc9=shift(Muc9, nshift)
Muc10=shift(Muc10, nshift)




#pour dessiner le signal condensé, tronqué shifté avec N points de mesures.
# on transforme la simulation de mesure complexe en réels
#Ici k est la mesure du temps en cs
Muc1=[Muc1[k].real for k in range(N)]
Muc2=[Muc2[k].real for k in range(N)]
Muc3=[Muc3[k].real for k in range(N)]
Muc4=[Muc4[k].real for k in range(N)]
Muc5=[Muc5[k].real for k in range(N)]
Muc6=[Muc6[k].real for k in range(N)]
Muc7=[Muc7[k].real for k in range(N)]
Muc8=[Muc8[k].real for k in range(N)]
Muc9=[Muc9[k].real for k in range(N)]
Muc10=[Muc10[k].real for k in range(N)]

#______________________________________________________________________________
#          FIN DE LA CONDENSATION DES SIMULATIONS DES MESURES 
#______________________________________________________________________________


#______________________________________________________________________________
#                            TRACES DES SIMULATIONS
#_____________________________________________________________________________


figure1=plt.figure(1,figsize=(20,6))

#on trace d'abord le signal simulé avec NI2 points.
# L'abcisse est en cs 
plt.subplot(1,2,1) 

plt.plot(t, tabI31, 'm',lw=2, label=capteur1)
plt.plot(t, tabI32, 'b',lw=2, label=capteur2)
plt.plot(t, tabI33, 'c',lw=2, label=capteur3)
plt.plot(t, tabI34, 'g',lw=2, label=capteur4)
plt.plot(t, tabI35, 'y',lw=2, label=capteur5)
plt.plot(t, tabI36, 'r',lw=2, label=capteur6)
plt.plot(t, tabI37, 'm--',lw=2, label=capteur7)
plt.plot(t, tabI38, 'b--',lw=2, label=capteur8)
plt.plot(t, tabI39, 'c--',lw=2, label=capteur9)
plt.plot(t, tabI40, 'g--',lw=2, label=capteur10)
plt.legend(loc='best')
plt.title('Signal simulé,  Nombre de points='+str(NI2))



#on trace ensuite le signal condensé tronqué shifté avec N points.
plt.subplot(1,2,2)

plt.plot(Muc1, 'm',lw=2, label=capteur1)
plt.plot(Muc2, 'b',lw=2, label=capteur2)
plt.plot(Muc3, 'c',lw=2, label=capteur3)
plt.plot(Muc4, 'g',lw=2, label=capteur4)
plt.plot(Muc5, 'y',lw=2, label=capteur5)
plt.plot(Muc6, 'r',lw=2, label=capteur6)
plt.plot(Muc7, 'm--',lw=2, label=capteur7)
plt.plot(Muc8, 'b--',lw=2, label=capteur8)
plt.plot(Muc9, 'c--',lw=2, label=capteur9)
plt.plot(Muc10, 'g--',lw=2, label=capteur10)
plt.title('Signal après troncature et shift,  Nombre de points='+str(N))
plt.xlabel('cut1= '+str(cut1) +'  cut2= '+str(cut2)+ '  shift= '+str(nshift))
plt.legend(loc='best')
plt.show()








###############################################################################
#               FIN DE LA SIMULATION D'UN PASSAGE DE CAMION
################################################################################
#Ce premier chapitre sera à remplacer par l'importation de mesures réelles 
#lorsque nous voudrons tester l'efficacité des algorythmes ci dessous pour 
#déterminer la primitive du signal camion réel











#%% CELLULE 2 %%#

###############################################################################
#               RECONSTRUCTION DE LA PRIMITIVE D'UN SIGNAL CAMION
################################################################################
#En faisant l'hypothèse que le pont a un comportement linéaire lors du passage 
#des camions, on montre que les signaux de mesures s'obtiennent par convolution 
#du signal camion avec la fonction d'influence d'un essieu.
#La résolution de cette équation de convolution peut etre faite à l'aide 
#de transformées de Fourier.
#Les signaux de mesures étant discrétisés au pas d'un centiseconde, et la 
#fonction d'influence étant discrétisée au pas d'un centimètre, on peut espérer
#que l'utilisation de transformées de Fourierdiscrète (TFD) et de la convolution
#circulaire permettra d'approcher la détermination du signal camion
#ou de sa primitive
#Nous utilisons ci dessous deux algorithmes pour ce faire.


#______________________________________________________________________________
#          DETERMINATION DE L'ESTIMATION DE LA VITESSE Vprime
#______________________________________________________________________________
#vitesse de l'essieu en m/s ou cm/cs (c'est un entier)


# Nous allons estimer la vitesse de deux manières.
# 1/ Nous "mesurons le temps écoulé entre les pics observés sur les capteurs 
# N°2 (CO-T2-P4) et N°5(CO-T3-P4). Nous faisons l'approximation que le pic est 
# observé au moment du passage du camion au dessus du capteur.
# Chacun de ces capteurs est positionné au milieu de la travée correspondante. 
#Nous connaissons donc la distance L0 parcourue pendant cet intervale de temps.
# la vitesse et supposée constante égale à V5_2, (approximée par Va5_2) .
# 2/ nous faisons la même chose avec le couple de capteur 
#N°3 (CO-T2-P7) et N°6 (CO-T3-P7) pour obtenir l'estimation V6_3
#(approximée par Va6_3).
# On a les instants auxquels ces pics sont atteints (argmax(Mi)), i=2,5,3,6)
# et l'on fait la différence pour chaque couple de capteur. (dpicMi_j)
# Si (dpicMi_j>0) cela veut dire que le véhicule roule de T2 à T3 
#(Le capteur Mi est sous la travée T2 et Mj est sous la travée T3)
# C'est bien sûr le contraire si (dpicMi_j<0)
# dpicMi_j est le temps de transit entre le milieu d'une travée et le milieu
# de l'autre en centiseconde. 
#Comme cette distance L0 est connue (1460cm) on a une estimation de la vitesse.
#que l'on donne ci dessous en m/s (ou cm/cs)   

# Longieur de la travée T2 ou T3 en cm
# égale à la distance entre les capteurs 2 et 5 ou 3 et 6
L0=1460 

NpicM2=np.argmax(Muc2)
NpicM5=np.argmax(Muc5)
dpicM5_2=NpicM5-NpicM2
V5_2=L0/dpicM5_2
NpicM3=np.argmax(Muc3)
NpicM6=np.argmax(Muc6)
dpicM6_3=NpicM6-NpicM3
V6_3=L0/dpicM6_3
# print("NpicM2",NpicM2)
# print("NpicM5",NpicM5)
# print("dpicM5_2=",dpicM5_2)
print ("V5_2=",V5_2)
# print("NpicM3",NpicM3)
# print("NpicM6",NpicM6)
# print("dpicM6_3=",dpicM6_3)    
print ("V6_3=",V6_3)    

## Malheureusement on constate le plus souvent des estimations un peu différentes 
#pour des couples de capteurs différents.

# Le Script Python "20 12 06 Vitesse.py permet de tracer la comparaison de ces 
#estimations de vitesse pour différents passages 

#On retiendra V5_2 pour estimer la vitesse.
Vprime=int(V5_2)
#___________________________________________________________________________
#               FIN DE L'ESTIMATION DE LA VITESSE
#____________________________________________________________________________



#______________________________________________________________________________
#          DETERMINATION DE L'ESTIMATION DE L'ORDONNEE DE PASSAGE y_cprime
#______________________________________________________________________________
#ordonnée du centre de l'essieu en cm sur le tablier du pont. C'est un entier divisible par 10.
# y_cprime=210


# Pour le calcul des moyennes de signaux capteurs nous devons faire la somme Sc
# Commençons par initialiser Sc à 0
S1 = 0
S2 = 0
S3 = 0
S4 = 0
S5 = 0
S6 = 0
S7 = 0
S8 = 0
S9 = 0
S10 = 0

# calcul des sommes Sc     
for i in range(N):
    S1=S1+Muc1[i]
    S2=S2+Muc2[i]
    S3=S3+Muc3[i]
    S4=S4+Muc4[i]
    S5=S5+Muc5[i]
    S6=S6+Muc6[i]
    S7=S7+Muc7[i]
    S8=S8+Muc8[i]
    S9=S9+Muc9[i]
    S10=S10+Muc10[i]
# calcul des moyennes Moyc
Moy1=S1/N
Moy2=S2/N 
Moy3=S3/N
Moy4=S4/N 
Moy5=S5/N
Moy6=S6/N 
Moy7=S7/N
Moy8=S8/N 
Moy9=S9/N
Moy10=S10/N 

# print ("Moy1=",Moy1)
# print ("Moy2=",Moy2)
# print ("Moy3=",Moy3)
# print ("Moy4=",Moy4)
# print ("Moy5=",Moy5)
# print ("Moy6=",Moy6)
# print ("Moy7=",Moy7)
# print ("Moy8=",Moy8)
# print ("Moy9=",Moy9)
# print ("Moy10=",Moy10)

# Vérification du type de M1
#MM=type(Muc1)
#print ('MM=',MM)
#Réponse:MM= <class 'list'> 

#Pour déterminer l'ordonnée yc de la trajectoire du centre des essieux, 
#nous utilisons ce que nous avons nommé l'indice "mesuré" Hmi_j, 
#où i et j sont des numéros de capteurs positionnés symétriquement par rapport
# à l'axe du pont. (par exemple capteurs 4 et 1 ou capteurs 3 et 2)
#Cet indice se calcule à partir des moyennes des mesures des deux capteurs.
#Nous retenons les couple 3_2 (CO-T2-P7_CO-T2-P4) et 4_1 (CO-T2-P9_CO-T2-P1)



Hm4_1=(Moy4-Moy1)/(Moy4+Moy1)
Hm3_2=(Moy3-Moy2)/(Moy3+Moy2)
print ('Hm4_1=',Hm4_1)
print ('Hm3_2=',Hm3_2)



#L'approche de modélisation permet de montrer que l'indice "théorique" Hti_j 
#est une fonction impaire de yc qui est, avec une excellente approximation,
#approchée par un polynôme de degré 3 strictement croissant.
#Hti_j dépend aussi de l'écartement er des roues.
#
#                  Nous continuons dans le cas où er=180cm.
#
#Si on retient le couple de capteur 3_2 (CO-T2-P7_CO-T2-P4) Le script 
# "21 02 06 Influence.py", associé à la détermination de la courbe de tendance
# avec EXCEL, montre que 
# Ht3_2=-7,20E-09yc^3+3,16E-03yc.
#On doit avoir Ht3_2=Hm3_2.
#Ainsi à partir des mesures on a une équation de la forme x^3=px+q avec 

#D'après les formules de Cardan on peut donner la racine réelle de ce polynôme
#Pour cela on calcule



q=-int((Hm3_2/7.2)*1000000000)
print("q=",q)
p=int((3.16/7.2)*1000000)
print("p=",p)


yc3_2=(((q*q/4)-(p*p*p/27))**(1/2)+q/2)**(1/3)-(((q*q/4)-(p*p*p/27))**(1/2)-q/2)**(1/3)
print ("yc3_2=",yc3_2)


#Si ce nombre et réel c'est la racine recherchée.
#Si ce nombre est complexe, la racine réelle = -2*la partie réelle de yc3_2
#(Considérez les 3 racines cubiques d'un réel, 
#(une réelle et deux complexes conjuguées dont la somme=0))
#Déterminons d'abord la partie réelle de yc3_2



Ryc3_2=yc3_2.real
print ("re=", Ryc3_2)
#puis sa partie imaginaire
Iyc3_2=yc3_2.imag
print ("I=", Iyc3_2)
if Iyc3_2!=0:
    print ("il est complexe")
    yc3_2=-2*Ryc3_2
else:
    print ("il est réel")        
print('yc3_2=',yc3_2)




#un autre couple de capteurs symétriques donnerait une estimation de yc différente
#Si on retient le couple de capteur 4_1 (CO-T2-P9_CO-T2-P2),Le script 
# "21 02 06 Influence.py", associé à la détermination de la courbe de tendance
# avec EXCEL donne  Ht4_1=-3,15E-08yc^3+6,10E-03yc
#A partir des mesures on a encore une équation de la forme x^3=px+q avec


 
q=-int((Hm4_1/3.15)*100000000)
print("q=",q)
p=int((6.1/3.15)*100000)
print("p=",p)


#D'après les formule de Cardan on peut donner la racine réelle de ce polynôme
#Pour cela on calcule

yc4_1=(((q*q/4)-(p*p*p/27))**(1/2)+q/2)**(1/3)-(((q*q/4)-(p*p*p/27))**(1/2)-q/2)**(1/3)

#Si ce nombre et réel c'est la racine recherchée.
#Si ce nombre est complexe, la racine réelle = -2*la partie réelle de yc4_1



Ryc4_1=yc4_1.real
print ("re=", Ryc4_1)
Iyc4_1=yc4_1.imag
print ("I=", Iyc4_1)
if Iyc4_1!=0:
    print ("il est complexe")
    yc4_1=-2*Ryc4_1
else:
    print ("il est réel")        
print('yc4_1=',yc4_1)



#On constate généralement une différence pas complètement négligeable
# entre les deux déterminations de yc.
# (Voir script python "20 12 21 Comparaison yc4_1, ycv3_2.py"
# ou l'image "20 12 22 yc3_2 versus yc4_1.png")
#Il semble y avoir quatre raisons à cela.
#La première est que le polynôme de degré 3 associé à Hm4_1 est plus éloigné 
#que dans le cas Hm3_2 pour lequel la correspondance est quasi parfaite.
#Voir graphes EXCEL H(yc,er=180)
#La deuxième raison est que pour cette valeur de Hm4_1 on est dans une partie 
#"plateau" de la courbe et une petite variation de Hm4_1 donne une grande 
#variation de yc4_1
#La troisième raison est que le capteur 1 (CO-T2-P2) donne des signes d'un 
#défaut de symétrie par rapport au capteur 4 (CO-T2-P9)
# La quatrième raison est qu'il semble y avoir un petit défaut dans la 
#détermination des fonctions d'influence pour les capteurs 1 et 4.
#Dans la suite nous retiendrons yc3_2 comme estimation de yc
#y_cprime=round(yc3_2/10)*10
y_cprime=round(yc3_2/10)*10
print("y_cprime=",y_cprime)

SMALLER_SIZE =8
BIGGER_SIZE = 30
plt.rc( 'legend', fontsize=BIGGER_SIZE) 



plt.figure(2,figsize=(20,10))
plt.plot(0,0, 'b',lw=4, label="Vprime="+str(round(Vprime,2)))
plt.plot(0,0, 'm',lw=4, label="y_cprime="+str(round(y_cprime,2)))
plt.title("Estimation de la vitesse et de l' ordonnée de passage")

#plt.xlabel('cut1= '+str(cut1) +'  cut2= '+str(cut2)+ '  shift= '+str(nshift))
plt.legend(loc='center')
plt.show()
#
plt.rc( 'legend', fontsize=SMALLER_SIZE)

#___________________________________________________________________________
#                FIN DE L'ESTIMATION DE L'ORDONNEE DE PASSAGE
#________________________________________________________________

#______________________________________________________________________________
#          FIN DE LA CONDENSATION DES SIMULATIONS DES MESURES 
#_____________________________________________________________________________
#______________________________________________________________________________
#%% CELLULE 3 %%#

#______________________________________________________________________________
#          DETERMINATION DE L'ESTIMATION DE L'ECARTEMENT DES ROUES e_Rprime
#______________________________________________________________________________
#écartement entre les centres des roues de l'essieu en dm. C'est un entier pair
e_Rprime=18 

#On en déduit l'ordonnée de la roue gauche en cm (par rapport à la gauche du tablier) 
#(C'est un entier divisible par 10),
y_gprime=y_cprime-10*int(e_Rprime/2)
#l'ordonnée de la roue gauche en dm (par rapport à l'axe du tablier). 
#C'est un entier
nygprime=int((y_gprime/10)+44)
#et l'ordonnée de la roue droite en dm (par rapport à l'axe du tablier). 
#C'est un entier
nydprime=nygprime+e_Rprime


#On retrace le signal camion avec quelques informations complémentaires



#s3='CO-T2-P2'
s3_1='CO-T2-P2_filtre'
#s3= 'CO-T2-P7'
s3_2='CO-T2-P4_filtre'
#s3='CO-T2-P9'
s3_3='CO-T2-P7_filtre'
#s3='CO-T2-P4'
s3_4='CO-T2-P9_filtre'
#s3='CO-T3-P4'
s3_5='CO-T3-P4_filtre'
#s3='CO-T3-P7'
s3_6= 'CO-T3-P7_filtre'
#s3='CO-AXL-23'
s3_7='CO-AXL-23_filtre'
#s3="CO-AXL-45"
s3_8="CO-AXL-45_filtre"
#s3="CO-AXL-67"
s3_9="CO-AXL-67_filtre"
#s3='CO-AXL-89'
s3_10= 'CO-AXL-89_filtre'
# Commençons par définir le temps t (en centiseconde)
t= np.zeros(N)
for i in range(0,N):
    t[i] = i

# Préparons les Strings des titres des courbes avec la valeur moyenne du signal et les frecal
s4_1=s3_1+"    Moy= "+str(round(Moy1,5))
s4_2=s3_2+"    Moy= "+str(round(Moy2,5))
s4_3=s3_3+"    Moy= "+str(round(Moy3,5))  
s4_4=s3_4+"    Moy= "+str(round(Moy4,5)) 
s4_5=s3_5+"    Moy= "+str(round(Moy5,5)) 
s4_6=s3_6+"    Moy= "+str(round(Moy6,5)) 
s4_7=s3_7+"    Moy= "+str(round(Moy7,5)) 
s4_8=s3_8+"    Moy= "+str(round(Moy8,5)) 
s4_9=s3_9+"    Moy= "+str(round(Moy9,5)) 
s4_10=s3_10+"   Moy="+str(round(Moy10,5)) 

# Tracé des coubes des signaux
plt.figure(3,figsize=(20,10))
plt.plot(t,Muc1, 'm',lw=2,label=s4_1)
plt.plot(t,Muc2, 'b',lw=2,label=s4_2)
plt.plot(t,Muc3, 'c',lw=2,label=s4_3)
plt.plot(t,Muc4, 'g',lw=2,label=s4_4)
plt.plot(t,Muc5, 'y',lw=2,label=s4_5)
plt.plot(t,Muc6, 'r',lw=2,label=s4_6)
plt.plot(t,Muc7, 'm--',lw=2,label=s4_7)
plt.plot(t,Muc8, 'b--',lw=2,label=s4_8)
plt.plot(t,Muc9, 'c--',lw=2,label=s4_9)
plt.plot(t,Muc10, 'g--',lw=2,label=s4_10)

# Titre des axes 
#incluant sur l'axe x 
#les informations sur le nombre de points NM1 et les estimations des vitesses
#incluant sur l'axe y les estimations des ordonnées de passage yc3_2 et tc4_1
# et les paramètres de recalage des raideurs 

plt.xlabel(r'$t$ (cs)   '+" Nbe de points="+str(N)+"   V5_2="+str(round(V5_2,2))+"m/s  V6_3="+str(round(V6_3,2))+"m/s"+"   Vprime="+str(round(Vprime,2))+"m/s",size=18)
plt.ylabel('déf(mm/m) yc3_2='+str(int(yc3_2))+'cm yc4_1=' +str(int(yc4_1))+'cm y_cprime='+str(y_cprime),size=14)

# # Donnons le nom du fichier de mesure dans le titre en ajoutant la distance entre roues er
plt.title('Signal après troncature et shift,  Nombre de points='+str(N)+'   er=180',size=16)
plt.xlim([0,t[N-1]])
plt.legend(loc='best')
plt.grid(True);
plt.show()

#______________________________________________________________________________
#%% CELLULE 4 %%#

#_____________________________________________________________________________
#           DEFINITION DE L'OPERATEUR TRANSFORMEE DE FOURIER DISCRETE
#_____________________________________________________________________________

#Un calcul rapide de la transformée de Fourier discrète peut etre fait en utilisant 
# le produit du vecteur à transformer par une matrice dont les composantes 
#complexes ont toutes pour norme 1.
#La transformée inverse utilise alors la matrice dont les composantes sont 
#les conjugées des composantes de la matrice précédente. 
#Ce sont aussi les inverses puisque la norme des composantes est 1

#On définit ainsi une matrice M de taille N*N telle que 
#M[k][n]=exp(j*2*np.pi*k*n/N)
# N est déterminé au premier chapitre 
# (k indice de ligne de 0 à N-1, n indice de colonne de 0 à N-1)
#On peut faire mieux que calculer les exponentielles pour chaque composante.
#La ligne k=0 et la colonne n=0 sont remplies de 1. 
#On calcule le complexe z=exp(j*2*np.pi/N) et on le met dans la composante (1,1)
#On calcule la composante (k,1) par M[k][1]=M[k-1][1]*z ce qui fait bien z**n 
#avec un calcul plus rapide que celui d'une exponentielle.
#On continue jusqu'à k=N-1 (composante (N-1,1)). La deuxième ligne (k=1) est 
#ainsi remplie.
#Pour les autres lignes on rempli d'abord la composante de la diagonale 
#M[k][k]=M[k][k-1]*M[k][1], puis les composantes supèrieures à la diagonale
#M[k][n]=M[k][n-1]*M[k][1], avec n>k.
#Enfin on complète par symétrie les composantes inférieures à la diagonale (n<k).

#Pour connaitre le temps de calcul nécessaire pour construire cette matrice
#on démarre un chronomètre 
start_matrix = timeit.default_timer()
#On intialise la matrice N*N avec des 1 partout. 
#Les composantes sont des complexes de double longueur
M=np.ones((N,N),dtype = np.clongdouble)

#Calcul de z=exp(j*2*pi/N) (np.exp(complex(partie réelle, partie imaginaire)))
z=np.exp(complex(0,2*np.pi/N))
#print ("z=",z)
# Boucle sur k à partir de k=1 (2ème ligne) jusqu'à k=N-1 (Nème ligne)
for k in range(1,N): 
    if k==1:
        M[k][1]=z
    else:
        M[k][1]=M[k-1][1]*z
#Boucle sur n à partir de n=1 (2ème colonne) jusqu'à n=N-1
    for n in range(1,N): 
        if n>=k:
            M[k][n]=M[k][n-1]*M[k][1]
        else: #on complète parsymétrie
            M[k][n]=M[n][k]
#On arrète le chronomètre
stop_matrix = timeit.default_timer()
#On affiche la durée du calcul de M
tmatrix=stop_matrix - start_matrix
print('Temps calcul matrice: ', tmatrix ) 
#Par exemple pour N=800, le temps de calcul est de l'ordre de 0,5s.
#Pour N=8000 il est de l'ordre de 50s (le temps de calcul est en N*N) 
 

# La transformée de Fourier discrète du vecteur f est obtenue en multipliant 
#la matrice dont les composantes sont les conjugées (ou les inverses) de celles
# de M par le vecteur f.
#On le fait en construisant deux fonctions: 
#la première tilde(f,k) donne la transformée de Fourier de f au point k
#la deuxième tilde_list(f) donne le vecteur "transformée de Fourier de f"

#Transformée de Fourier d'une fonction f au point k
#On utilise ici np.dot qui calcul le produit tensoriels de deux arrays.
#Pour deux vecteurs, c'est le produit scalaire
#On calcul donc la somme de 0 à N-1 de f[n]M[k][n] 
def tilde(f,k): 
    inverseM=1/M[k] 
#C'est le vecteur dont les composantes sont les conjuguées des composantes 
#du vecteur M[k]    
    return np.dot(f,inverseM)

#Transformée de Fourier d'une fonction f
#On donne en entrée la liste des {f[k]}, on sort avec la liste des {fourier(f)[k]}
def tilde_list(f):
    tempo = np.zeros(N, dtype = np.clongdouble)
    for k in range(len(tempo)):
        tempo[k]=tilde(f,k)
    return tempo

#Pour la transformée de Fourier inverse, on multiplie par M, en n'oubliant 
#pas de diviser par N  
#On procède comme ci dessus  
#Transformée de fourier inverse d'une fonction au point k
def untilde(ftilde,k):
    return 1/N*np.dot(ftilde,M[k])

#Transformée de fourier inverse d'une fonction f 
#on donne en entrée la liste des {fourier(f)[k]}, on sort avec la liste des {f[k]} 
def untilde_list(ftilde):
    tempo = np.zeros(N, dtype = np.clongdouble)
    for k in range(len(tempo)):
        tempo[k]=untilde(f,k)
    return tempo

#_____________________________________________________________________________
#       FIN DE LA DEFINITION DE L'OPERATEUR TRANSFORMEE DE FOURIER DISCRETE
#_____________________________________________________________________________

#_____________________________________________________________________________
#           DEFINITION DE L'OPERATEUR DE CONVOLUTION CIRCULAIRE
#_____________________________________________________________________________

    
#rend h[n] = (fXg)[n]
def convolution(f,g,n):
    res=0
    for p in range(N):
        res+=f[n-p]*g[p]
    return res

#_____________________________________________________________________________
#           FIN DE LA DEFINITION DE L'OPERATEUR DE CONVOLUTION CIRCULAIRE
#_____________________________________________________________________________

#_____________________________________________________________________________
#    PROGRAMME DE RESOLUTION DE L'EQUATION DE CONVOLUTION CIRCULAIRE
#_____________________________________________________________________________

#Cette opérateur résoud l'équation (f conv h = g ) où f est inconnue et h et g connus
def res_conv_circ(h,g): 
    #On a des arrays contenant g et h, deux arrays à N éléments chacun 
    ftilde=[tilde(g,k)/tilde(h,k) for k in range(N)]
    f=[untilde(ftilde,k) for k in range(N)]
    return f

#_____________________________________________________________________________
#      FIN DU PROGRAMME DE RESOLUTION DE L'EQUATION DE CONVOLUTION CIRCULAIRE
#_____________________________________________________________________________




#Le problème que nous devons résoudre est celui de l'identification d'un camion
# à partir des mesures des différents capteurs positionnés sur le pont
#Cette identification fait appel à une fonction d'influence d'un essieu.
# Cette fonction d'influence peut etre déterminée par combinaison linéaire 
#de deux fonctions d'influence monocharge unitaire. 
#(une pour chaque roue de l'essieu chargée à 1 tonneforce).
#La fonction d'influence monocharge a été déterminée à partir d'un modèle 
#aux éléments finis de l'ouvrage.
#Nous devons donc cans un premier temps importer ces fonctions d'influence monocharge
# C'est l'objet du paragraphe ci dessous
#Nous devrons ensuite construire les fonctions d'influence d'un essieu 
#(dépendant de yc_prime et eR_prime)


#Ce travail a déjà été fait ci dessus pour simuler le passage d'un camion.
#Nous le refaisons ici de façon à rendre indépendants les deux chapitres de ce 
#script et de pouvoir remplacer le chapitre de simulation camion par un chapitre
#d'importation de données de mesures réelles, sans avoir à modifier le deuxième
#chapitre.

#______________________________________________________________________________
#       IMPORTATION DES FONCTIONS D'INFLUENCE MONOCHARGE UNITAIRE
#______________________________________________________________________________

#  Le string du chemin d'acces aux fonctions d'influence monocharge unitaire 
#calculées numériquement avec ASTER est 
s1="./Complet_Isym_ttcapt_V36.json"
#Importons toutes ces fonctions d'influence dans le fichier data1
f = open(s1)
data1 = json.load(f)
f.close()

# f est un dictionnaire.

# On peut demander les clefs de data1 par
#clefs=data1.keys()
#print("Clefs de data1",clefs)
#On obtient
#Clefs de data1 dict_keys(['CO_AXL_67', 'CO_AXL_45', 'CO_AXL_23', 'CO_AXL_89',
# 'CO_T2_P2', 'CO_T2_P4', 'CO_T2_P7', 'CO_T2_P9', 'CO_T3_P4', 'CO_T3_P7'])
#On voit qu'il ya 10 fichiers dans le dictionnaire.
#Ils se distinguent par capteurs 
#"Précisons qu'il s'agit des modèles des capteurs dans le modèle numérique du pont
#"Ces fichiers donnent la simulation de la mesure d'un capteur lorsqu'une charge
# unitaire de 1tf est placé en un point (x,y) du tablier
#Le modèle du pont est symétrique par rapport à un axe y=0.
#Les capteurs sont disposés symétriquement.
#Commençons par les renommer en changeant les srings qui les désignent
#Chaque capteur est associé à son symétrique. La notation est redondante
#Par exemple la couple capteur1,capteur1s est identique au couple capteur 4 capteur 4s
#Ceci est fait pour une raison de commodité dans la programation ci dessous.
capteur1= 'CO_T2_P2'
capteur1S='CO_T2_P9'
capteur2= 'CO_T2_P4'
capteur2S='CO_T2_P7'
capteur3= 'CO_T2_P7'
capteur3S='CO_T2_P4'
capteur4= 'CO_T2_P9'
capteur4S='CO_T2_P2'
capteur5= 'CO_T3_P4'
capteur5S='CO_T3_P7'
capteur6= 'CO_T3_P7'
capteur6S='CO_T3_P4'
capteur7= 'CO_AXL_23'
capteur7S='CO_AXL_89'
capteur8= 'CO_AXL_45'
capteur8S='CO_AXL_67'
capteur9= 'CO_AXL_67'
capteur9S='CO_AXL_45'
capteur10= 'CO_AXL_89'
capteur10S='CO_AXL_23'

#data1[capteurk] est lui même un dictionnaire dont on peut demander les clefs
#clefs=data1[capteurk].keys()
#print("Clefs de data1 bis",clefs)
#On obtient
#Clefs de data1 bis dict_keys(['0_0', '0_1', '0_2', '0_3', '0_4', '0_5', '0_6',
# '0_7', '0_8', '0_9', '1_0', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', 
# '1_7', '1_8', '1_9', '2_0', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', 
# '2_7', '2_8', '2_9', '3_0', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', 
# '3_7', '3_8', '3_9', '4_0', '4_1', '4_2', '4_3', '4_4'])
# On voit qu'il ya 45 fichiers dans le dictionnaire.
#Ils se distinguent par la valeur de y 
#(une valeur de y tous les 10cm de 0 à 440). 
#A y en cm on associe tout d'abord l'entier ny=int(y/10)
#Puis à une valeur de ny est associé le string (str(a)+"_"+str(b)) 
#avec a=int(ny/10) et b=int(ny%10)

#data1[capteurk][clefy] est lui même un dictionnaire dont on peut demander les clefs
#clefs=data1[capteurk][clefy1].keys()
#print("Clefs de data1 ter",clefs)
#On obtient 2 clefs de data1 ter dict_keys(['inst', capteurk+'C_dyn_moda'])
#Le premier tableau 'inst' correspond aux instants de simulation de la 
#fonction d'influence avec une vitesse simulée de 36 km/h, toutes les 0,001s.
#Le deuxième tableau est la valeur de la fonction d'influence 
#pour le capteur 'capteurk' et la valeur de y correspondant à la clef 'clefy1'.
# La fonction d'influence est fonction de x avec un point par cm
#( 36km/h=10m/s et la période d'acqusition dans 'inst' est 0,001s).
#Il y a 4591 valeurs pour chaque fichier correspondant à la longueur de 4590cm du pont.

#Construisons les strings de clefy 
clefy=[]
for ny in range(45):
    a=int(ny/10)
    b=int(ny%10)
    clefy.append(str(a)+"_"+str(b))
#Construisons les strings des 10 capteurs et de leur symétrique
tableau1=capteur1+"_dyn_moda"
tableau1S=capteur1S+"_dyn_moda"
tableau2=capteur2+"_dyn_moda"
tableau2S=capteur2S+"_dyn_moda"
tableau3=capteur3+"_dyn_moda"
tableau3S=capteur3S+"_dyn_moda"
tableau4=capteur4+"_dyn_moda"
tableau4S=capteur4S+"_dyn_moda"
tableau5=capteur5+"_dyn_moda"
tableau5S=capteur5S+"_dyn_moda"
tableau6=capteur6+"_dyn_moda"
tableau6S=capteur6S+"_dyn_moda"
tableau7=capteur7+"_dyn_moda"
tableau7S=capteur7S+"_dyn_moda"
tableau8=capteur8+"_dyn_moda"
tableau8S=capteur8S+"_dyn_moda"
tableau9=capteur9+"_dyn_moda"
tableau9S=capteur9S+"_dyn_moda"
tableau10=capteur10+"_dyn_moda"
tableau10S=capteur10S+"_dyn_moda"
#Préparons 10 tableaux à 89 lignes et 4591 colonnes.
#Les 89 lignes correspondent à 89 valeurs de y de y=-440 à +440 
#avec une valeur tous les 10 cm.
#Les 4591 colonnes correspondent à une colonne tous les cm en x.
#Chaque tableau correspond à un capteur (tabk correspond au capteur k, k=1à10) 
lignes, colonnes=89, 4591
tab1=[[0]*colonnes]*lignes
tab2=[[0]*colonnes]*lignes
tab3=[[0]*colonnes]*lignes
tab4=[[0]*colonnes]*lignes
tab5=[[0]*colonnes]*lignes
tab6=[[0]*colonnes]*lignes
tab7=[[0]*colonnes]*lignes
tab8=[[0]*colonnes]*lignes
tab9=[[0]*colonnes]*lignes
tab10=[[0]*colonnes]*lignes


for ny in range(45):
#y=0 correspond à l'indice 44 pour les composantes de tabk
#nny est un indice pour les valeurs de y positives ou nulles. Il varie de 44 à 88.
#y=(nny-44)*10
#my est l'indice pour les valeurs de y négatives ou nulles. Il varie de 0 à 44.
#y=(my-44)*10. (l'indice 44 est vu deux fois, mais tous les deux correspondent à y=0)
    nny=ny+44
    my=44-ny
# Les tabk (k=1 à 10) sont remplis en faisant appels aux données des  
#deux capteurs symétriques capteurk et capteurkS   
    tab1[nny]=data1[capteur1][clefy[ny]][tableau1]
    tab1[my]=data1[capteur1S][clefy[ny]][tableau1S]
    tab2[nny]=data1[capteur2][clefy[ny]][tableau2]
    tab2[my]=data1[capteur2S][clefy[ny]][tableau2S]
    tab3[nny]=data1[capteur3][clefy[ny]][tableau3]
    tab3[my]=data1[capteur3S][clefy[ny]][tableau3S]
    tab4[nny]=data1[capteur4][clefy[ny]][tableau4]
    tab4[my]=data1[capteur4S][clefy[ny]][tableau4S]
    tab5[nny]=data1[capteur5][clefy[ny]][tableau5]
    tab5[my]=data1[capteur5S][clefy[ny]][tableau5S]
    tab6[nny]=data1[capteur6][clefy[ny]][tableau6]
    tab6[my]=data1[capteur6S][clefy[ny]][tableau6S]
    tab7[nny]=data1[capteur7][clefy[ny]][tableau7]
    tab7[my]=data1[capteur7S][clefy[ny]][tableau7S]
    tab8[nny]=data1[capteur8][clefy[ny]][tableau8]
    tab8[my]=data1[capteur8S][clefy[ny]][tableau8S]
    tab9[nny]=data1[capteur9][clefy[ny]][tableau9]
    tab9[my]=data1[capteur9S][clefy[ny]][tableau9S]
    tab10[nny]=data1[capteur10][clefy[ny]][tableau10]
    tab10[my]=data1[capteur10S][clefy[ny]][tableau10S]

#Il faut maintenant prolonger le tableau suivant les x...

#Prolongation des fichiers (NI1 -> NI2)
Tabres=[0]*(NI2-NI1)
for ny in range(lignes):
    tab1[ny]=tab1[ny]+Tabres
    tab2[ny]=tab2[ny]+Tabres
    tab3[ny]=tab3[ny]+Tabres
    tab4[ny]=tab4[ny]+Tabres
    tab5[ny]=tab5[ny]+Tabres
    tab6[ny]=tab6[ny]+Tabres
    tab7[ny]=tab7[ny]+Tabres
    tab8[ny]=tab8[ny]+Tabres
    tab9[ny]=tab9[ny]+Tabres
    tab10[ny]=tab10[ny]+Tabres

#tabc[ny][nx] donne la mesure du capteur c lorsque l'on met une charge unitaire
# sur le tablier du pont au point d'abcisse nx et d'ordonnée ny
#______________________________________________________________________________
#       FIN DE L'IMPORTATION DES FONCTIONS D'INFLUENCE MONOCHARGE UNITAIRE
#______________________________________________________________________________

#______________________________________________________________________________
#       CONSTRUCTION DES FONCTIONS D'INFLUENCE D'UN ESSIEU
#______________________________________________________________________________
#Les fonctions d'influence Jc d'un essieu sont des vecteurs à N composantes complexes 
#(pour utiliser les TFD)
#N a été déterminé au premier chapitre par N=round((L+Lmax)/(Vprime*dt))
#En cas de mesures réelles, il sera déterminé àpartir de la longueur de l'enregistrement
#
#Les Jc sont la combinaison linéaire  d'une 1/2 charge unitaire placée en 
#y=nyg et x=int(n*V) (roue gauche, 500kg) et d'une 1/2 charge unitaire placée en 
#y=nyd et x=int(n*V) (roue droite, 500kg)
#Le nombre de points de la fonction d'inflence monocharge est égal à L+Lmax
#Le nombre de points de la fonction d'influence d'un essieu est  N
#En général N<<L+Lmax sauf si Vprime =1 (dt=1)
#Par exemple pour NI2=L+Lmax=8000, et Vprime=10cm/cs, N=800.




# Préparons les vecteurs Jc1 àJc10 en les initailisant à 0 complexe de double longueur
Jc1=np.zeros((N), dtype = np.clongdouble)
Jc2=np.zeros((N), dtype = np.clongdouble)
Jc3=np.zeros((N), dtype = np.clongdouble)
Jc4=np.zeros((N), dtype = np.clongdouble)
Jc5=np.zeros((N), dtype = np.clongdouble)
Jc6=np.zeros((N), dtype = np.clongdouble)
Jc7=np.zeros((N), dtype = np.clongdouble)
Jc8=np.zeros((N), dtype = np.clongdouble)
Jc9=np.zeros((N), dtype = np.clongdouble)
Jc10=np.zeros((N), dtype = np.clongdouble)
# Remplissons les composantes des vecteurs Jc.
#tempo est le nx des fonctions d'influence monocharge correspondant à l'indice k dans le vecteur Jc
#Le coefficient 1000 ci dessous correspond à une différence d'unité 
#entre les fonctions d'influence monocharge et les fonctions d'influence essieu
#Notons que la fonction d'influence monocharge dépend de ny et de nx seulement
#alors que la fonction d'influence de l'essieu dépend de yc_prime et eR_prime via 
#nygprime et nydprime
for k in range(N):        
    tempo=int(k*dt*Vprime)
    Jc1[k]=1/2*(tab1[nygprime][tempo]+tab1[nydprime][tempo])*1000
    Jc2[k]=1/2*(tab2[nygprime][tempo]+tab2[nydprime][tempo])*1000
    Jc3[k]=1/2*(tab3[nygprime][tempo]+tab3[nydprime][tempo])*1000
    Jc4[k]=1/2*(tab4[nygprime][tempo]+tab4[nydprime][tempo])*1000
    Jc5[k]=1/2*(tab5[nygprime][tempo]+tab5[nydprime][tempo])*1000
    Jc6[k]=1/2*(tab6[nygprime][tempo]+tab6[nydprime][tempo])*1000
    Jc7[k]=1/2*(tab7[nygprime][tempo]+tab7[nydprime][tempo])*1000
    Jc8[k]=1/2*(tab8[nygprime][tempo]+tab8[nydprime][tempo])*1000
    Jc9[k]=1/2*(tab9[nygprime][tempo]+tab9[nydprime][tempo])*1000
    Jc10[k]=1/2*(tab10[nygprime][tempo]+tab10[nydprime][tempo])*1000
# modifions le dernier terme pour éviter une première composante nulle dans 
#le vecteur "Transformée de Fourier discrète" de la "Dérivée discrète" de Jc
#En effet dJc1tilde[0]=Jc1[N-1]-Jc1[0]
# Il faudra étudier les conséquences du choix de cette modification
Jc1[N-1]=Jc1[N-1]+0.000000001
Jc2[N-1]=Jc2[N-1]+0.000000001
Jc3[N-1]=Jc3[N-1]+0.000000001
Jc4[N-1]=Jc4[N-1]+0.000000001
Jc5[N-1]=Jc5[N-1]+0.000000001
Jc6[N-1]=Jc6[N-1]+0.000000001
Jc7[N-1]=Jc7[N-1]+0.000000001
Jc8[N-1]=Jc8[N-1]+0.000000001
Jc9[N-1]=Jc9[N-1]+0.000000001
Jc10[N-1]=Jc10[N-1]+0.000000001

#______________________________________________________________________________
#       FIN DE LA CONSTRUCTION DES FONCTIONS D'INFLUENCE D'UN ESSIEU
#______________________________________________________________________________



#______________________________________________________________________________
#       CALCUL DE LA DERIVEE DISCRETE DE LA FONCTION D'INFLUENCE ESSIEU
#______________________________________________________________________________
#Pour une deuxième construction du vecteur primitive signal camion directement
# nous construisons la "dérivée discrète" de Jc
# Préparons les vecteurs dJc1 à dJc10 en les initailisant à 0 complexe de double longueur
dJc1=np.zeros((N), dtype = np.clongdouble)
dJc2=np.zeros((N), dtype = np.clongdouble)
dJc3=np.zeros((N), dtype = np.clongdouble)
dJc4=np.zeros((N), dtype = np.clongdouble)
dJc5=np.zeros((N), dtype = np.clongdouble)
dJc6=np.zeros((N), dtype = np.clongdouble)
dJc7=np.zeros((N), dtype = np.clongdouble)
dJc8=np.zeros((N), dtype = np.clongdouble)
dJc9=np.zeros((N), dtype = np.clongdouble)
dJc10=np.zeros((N), dtype = np.clongdouble)
# Remplissons les composantes des vecteurs dJc.

for k in range(N-1): 
    dJc1[k]=Jc1[k+1]-Jc1[k]
    dJc2[k]=Jc2[k+1]-Jc2[k]
    dJc3[k]=Jc3[k+1]-Jc3[k]
    dJc4[k]=Jc4[k+1]-Jc4[k]
    dJc5[k]=Jc5[k+1]-Jc5[k]
    dJc6[k]=Jc6[k+1]-Jc6[k]
    dJc7[k]=Jc7[k+1]-Jc7[k]
    dJc8[k]=Jc8[k+1]-Jc8[k]
    dJc9[k]=Jc9[k+1]-Jc9[k]
    dJc10[k]=Jc10[k+1]-Jc10[k] 

#______________________________________________________________________________
#       FIN DU CALCUL DE LA DERIVEE DISCRETE DE LA FONCTION D'INFLUENCE ESSIEU
#______________________________________________________________________________
      




#______________________________________________________________________________
#          CALCUL DU SIGNAL CAMION
#______________________________________________________________________________
#Ce signal dans le cas idéal est une somme de Dirac, chacun correspondant à un essieu.
#Ce sont pour chaque capteur des vecteurs à N composantes complexes

#Initialisons à zéro complexe de longueur double ces vecteurs
p1=np.zeros(N, dtype= np.clongdouble)
p2=np.zeros(N, dtype= np.clongdouble)
p3=np.zeros(N, dtype= np.clongdouble)
p4=np.zeros(N, dtype= np.clongdouble)
p5=np.zeros(N, dtype= np.clongdouble)
p6=np.zeros(N, dtype= np.clongdouble)
p7=np.zeros(N, dtype= np.clongdouble)
p8=np.zeros(N, dtype= np.clongdouble)
p9=np.zeros(N, dtype= np.clongdouble)
p10=np.zeros(N, dtype= np.clongdouble)

#Chaque vecteur est la solution d'une équation de convolution circulaire

#Démarrons un chronomètre
start_solution = timeit.default_timer()

p1=res_conv_circ(Jc1,Muc1)
p2=res_conv_circ(Jc2,Muc2)
p3=res_conv_circ(Jc3,Muc3)
p4=res_conv_circ(Jc4,Muc4)
p5=res_conv_circ(Jc5,Muc5)
p6=res_conv_circ(Jc6,Muc6)
p7=res_conv_circ(Jc7,Muc7)
p8=res_conv_circ(Jc8,Muc8)
p9=res_conv_circ(Jc9,Muc9)
p10=res_conv_circ(Jc10,Muc10)


#On peut se contenter de garder la partie réel car la partie imaginaire doit etre très faible
#On peut en fait  vérifier que pc.imag=O(1e-13) comme prévu
p1=[p1[k].real for k in range(N)]
p2=[p2[k].real for k in range(N)]
p3=[p3[k].real for k in range(N)]
p4=[p4[k].real for k in range(N)]
p5=[p5[k].real for k in range(N)]
p6=[p6[k].real for k in range(N)]
p7=[p7[k].real for k in range(N)]
p8=[p8[k].real for k in range(N)]
p9=[p9[k].real for k in range(N)]
p10=[p10[k].real for k in range(N)]


#Pour l'affichage des résultats il est préférable de calculer la primitive 
#du signal camion qui dans le cas idéal est une foncttion en escalier

#Notons psumpc[n] cette primitive, c'est à dire la somme des valeurs prises 
#par le capteur jusqu'à n 

psump1=np.zeros(N)
psump2=np.zeros(N)
psump3=np.zeros(N)
psump4=np.zeros(N)
psump5=np.zeros(N)
psump6=np.zeros(N)
psump7=np.zeros(N)
psump8=np.zeros(N)
psump9=np.zeros(N)
psump10=np.zeros(N)
mpsump=np.zeros(N)


kk=np.zeros(N)

# Calculons psumpc

for k in range(N):
    psump1[k]=psump1[k-1]+p1[k]
    psump2[k]=psump2[k-1]+p2[k]
    psump3[k]=psump3[k-1]+p3[k]
    psump4[k]=psump4[k-1]+p4[k]
    psump5[k]=psump5[k-1]+p5[k]
    psump6[k]=psump6[k-1]+p6[k]
    psump7[k]=psump7[k-1]+p7[k]
    psump8[k]=psump8[k-1]+p8[k]
    psump9[k]=psump9[k-1]+p9[k]
    psump10[k]=psump10[k-1]+p10[k]
    mpsump[k]=(psump1[k]+psump2[k]+psump3[k]+psump4[k]+psump5[k]+psump6[k]+psump7[k]+psump8[k]+psump9[k]+psump10[k])/10
    kk[k]=k

   
#Notons ppsumpc[n] la primitive, calculée directement avec dJc 


# Initialisons les vecteurs à N composantes ppsumpc à 0
ppsump1=np.zeros(N, dtype= np.clongdouble)
ppsump2=np.zeros(N, dtype= np.clongdouble)
ppsump3=np.zeros(N, dtype= np.clongdouble)
ppsump4=np.zeros(N, dtype= np.clongdouble)
ppsump5=np.zeros(N, dtype= np.clongdouble)
ppsump6=np.zeros(N, dtype= np.clongdouble)
ppsump7=np.zeros(N, dtype= np.clongdouble)
ppsump8=np.zeros(N, dtype= np.clongdouble)
ppsump9=np.zeros(N, dtype= np.clongdouble)
ppsump10=np.zeros(N, dtype= np.clongdouble)


pp1=np.zeros(N)
pp2=np.zeros(N)
pp3=np.zeros(N)
pp4=np.zeros(N)
pp5=np.zeros(N)
pp6=np.zeros(N)
pp7=np.zeros(N)
pp8=np.zeros(N)
pp9=np.zeros(N)
pp10=np.zeros(N)
kk=np.zeros(N)

# Calculons ppsumpc

ppsump1=res_conv_circ(dJc1,Muc1)
ppsump2=res_conv_circ(dJc2,Muc2)
ppsump3=res_conv_circ(dJc3,Muc3)
ppsump4=res_conv_circ(dJc4,Muc4)
ppsump5=res_conv_circ(dJc5,Muc5)
ppsump6=res_conv_circ(dJc6,Muc6)
ppsump7=res_conv_circ(dJc7,Muc7)
ppsump8=res_conv_circ(dJc8,Muc8)
ppsump9=res_conv_circ(dJc9,Muc9)
ppsump10=res_conv_circ(dJc10,Muc10)


#On peut se contenter de garder la partie réel car la partie imaginaire doit etre très faible

pp1=[ppsump1[k].real for k in range(N)]
pp2=[ppsump2[k].real for k in range(N)]
pp3=[ppsump3[k].real for k in range(N)]
pp4=[ppsump4[k].real for k in range(N)]
pp5=[ppsump5[k].real for k in range(N)]
pp6=[ppsump6[k].real for k in range(N)]
pp7=[ppsump7[k].real for k in range(N)]
pp8=[ppsump8[k].real for k in range(N)]
pp9=[ppsump9[k].real for k in range(N)]
pp10=[ppsump10[k].real for k in range(N)]


#Ce calcul donne la primitive du signal camion à une fonction afine près.
#On peut corriger la fonction afine.
#Le vecteur corrigé est noté ppc..
ppc1=np.zeros(N)
ppc2=np.zeros(N)
ppc3=np.zeros(N)
ppc4=np.zeros(N)
ppc5=np.zeros(N)
ppc6=np.zeros(N)
ppc7=np.zeros(N)
ppc8=np.zeros(N)
ppc9=np.zeros(N)
ppc10=np.zeros(N)
mppc=np.zeros(N)

for k in range(N):
    

    ppc1[k]=pp1[k]-pp1[0]-(pp1[N-1]-pp1[N-int(N/3)])*k/(int(N/3))    
    ppc2[k]=pp2[k]-pp2[0]-(pp2[N-1]-pp2[N-int(N/3)])*k/(int(N/3)) 
    ppc3[k]=pp3[k]-pp3[0]-(pp3[N-1]-pp3[N-int(N/3)])*k/(int(N/3))    
    ppc4[k]=pp4[k]-pp4[0]-(pp4[N-1]-pp4[N-int(N/3)])*k/(int(N/3)) 
    ppc5[k]=pp5[k]-pp5[0]-(pp5[N-1]-pp5[N-int(N/3)])*k/(int(N/3)) 
    ppc6[k]=pp6[k]-pp6[0]-(pp6[N-1]-pp6[N-int(N/3)])*k/(int(N/3)) 
    ppc7[k]=pp7[k]-pp7[0]-(pp7[N-1]-pp7[N-int(N/3)])*k/(int(N/3))  
    ppc8[k]=pp8[k]-pp8[0]-(pp8[N-1]-pp8[N-int(N/3)])*k/(int(N/3)) 
    ppc9[k]=pp9[k]-pp9[0]-(pp9[N-1]-pp9[N-int(N/3)])*k/(int(N/3)) 
    ppc10[k]=pp10[k]-pp10[0]-(pp10[N-1]-pp10[N-int(N/3)])*k/(int(N/3))  
    mppc[k]=(ppc1[k]+ppc2[k]+ppc3[k]+ppc4[k]+ppc5[k]+ppc6[k]+ppc7[k]+ppc8[k]+ppc9[k]+ppc10[k])/10


#Arrètons le chronomètre et imprimons le temps nécessaire à la résolution 
#de l'équation de convolution

stop_solution = timeit.default_timer()
tsolution=stop_solution - start_solution
print('Temps de résolution= ', tsolution )

#______________________________________________________________________________
#          FIN DU CALCUL DU SIGNAL CAMION
#______________________________________________________________________________

#______________________________________________________________________________
#          TRACE DES DIFFERENTES FIGURES
#______________________________________________________________________________

#Les figures 6 et 30 représentent la moyenne sur les 10 capteurs des primitives
#des signaux camions reconstitués

plt.figure(4,figsize=(20,10))



#Les figures suivantes présentent la reconstruction du signal camion après 
#résolution de l'équation de convolution circulaire et sa "primitive"
#calculée par intégration ou directement
plt.subplot(3,4,1) 
plt.plot(p1, 'b',lw=2, label='signal camion')
plt.plot(psump1, 'g',lw=2, label='primitive ')
plt.plot(ppc1, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_T2_P2'+' s='+str(round(np.sum(p1),6)))
#plt.xlabel('somme CO_T2_P2= '+str(np.sum(p1))+' dt= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,2) 
plt.plot(p2, 'b',lw=2, label='signal camion')
plt.plot(psump2, 'g',lw=2, label='primitive ')
plt.plot(ppc2, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_T2_P4'+' s='+str(round(np.sum(p2),6)))
#plt.xlabel('somme CO_T2_P4= '+str(np.sum(p2))+' dt= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,3) 
plt.plot(p3, 'b',lw=2, label='signal camion')
plt.plot(psump3, 'g',lw=2, label='primitive ')
plt.plot(ppc3, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_T2_P7'+' s='+str(round(np.sum(p3),6)))
#plt.xlabel('somme CO_T2_P7= '+str(np.sum(p3))+' dt= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,4) 
plt.plot(p4, 'b',lw=2, label='signal camion')
plt.plot(psump4, 'g',lw=2, label='primitive ')
plt.plot(ppc4, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_T2_P9'+" s="+str(round(np.sum(p4),6)))
#plt.xlabel('somme CO_T2_P9= '+str(np.sum(p4))+' dt(cs)= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,5) 
plt.plot(p5, 'b',lw=2, label='signal camion')
plt.plot(psump5, 'g',lw=2, label='primitive ')
plt.plot(ppc5, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_T3_P4'+' s='+str(round(np.sum(p5),6)))
#plt.xlabel('somme CO_T3_P4= '+str(np.sum(p5))+' dt(cs)= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,6) 
plt.plot(p6, 'b',lw=2, label='signal camion')
plt.plot(psump6, 'g',lw=2, label='primitive ')
plt.plot(ppc6, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_T3_P7'+' s='+str(round(np.sum(p6),6)))
#plt.xlabel('somme CO_T3_P7= '+str(np.sum(p6))+' dt(cs)= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,7) 
plt.plot(p7, 'b',lw=2, label='signal camion')
plt.plot(psump7, 'g',lw=2, label='primitive ')
plt.plot(ppc7, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_AXL_23'+' s= '+str(round(np.sum(p7),6)))
#plt.xlabel('somme CO_AXL_23= '+str(np.sum(p7))+' dt(cs)= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,8) 
plt.plot(p8, 'b',lw=2, label='signal camion')
plt.plot(psump8, 'g',lw=2, label='primitive ')
plt.plot(ppc8, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_AXL_45'+' s='+str(round(np.sum(p8),6)))
#plt.xlabel('somme CO_AXL_45= '+str(np.sum(p8))+' dt(cs)= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,9) 
plt.plot(p9, 'b',lw=2, label='signal camion')
plt.plot(psump9, 'g',lw=2, label='primitive ')
plt.plot(ppc9, 'r',lw=2, label='primitive directement')
plt.title('capteur CO_AXL_67'+' s='+str(round(np.sum(p9),6)))
#plt.xlabel('somme CO_AXL_67= '+str(np.sum(p9))+' dt(cs)= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,10) 
plt.plot(p10, 'b',lw=2, label='signal camion')
plt.plot(psump10, 'g',lw=2, label='primitive ')
plt.plot(ppc10, 'r',lw=2, label='primitive directement')
plt.title('CO_AXL_89'+' s='+str(round(np.sum(p10),6)))
#plt.xlabel('somme CO_AXL_89= '+str(np.sum(p10))+' dt(cs)= '+ str(dt))
plt.grid()
plt.legend(loc='best')

plt.subplot(3,4,11) 


plt.plot(0,0, 'b',lw=4, label="Temps calcul matrice="+str(round(tmatrix,2)))
plt.plot(0,0, 'm',lw=4, label="Temps de résolution="+str(round(tsolution,2)))
plt.title("Estimation de la vitesse et de l' ordonnée de passage")

plt.legend(loc='center')


plt.show()     
#%% CELLULE 5 %%#

fig=plt.figure(5,figsize=(16,6))

##----- Figure 6 -----##
ax1 = fig.add_subplot(121)
ppp1 = ax1.plot(mpsump, 'k')
plt.title('moyenne primitive ')
plt.xlabel(' dt= '+ str(dt))
plt.grid()

##----- Figure 30 -----##
ax2 = fig.add_subplot(122)
ppp2 = ax2.plot(mppc, 'k')
plt.title('moyenne primitive directe')
plt.xlabel(' dt(cs)= '+ str(dt))
plt.grid()



plt.show()


