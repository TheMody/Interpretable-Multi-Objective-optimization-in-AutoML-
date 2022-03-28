from utils import delta,deltareverse, sigmoid, time_sigmoid, time_score ,score, score_scaled
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
Big_C = 10

def plotpareto3D(metric1, metric2,metric3, savepath = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(metric3)< len(metric1):
        metric1 = metric1[len(metric1)-len(metric3):]
        metric2 = metric2[len(metric2)-len(metric3):]
    #print(len(metric1), len(metric2), len(metric3))
    color= [(0.7* a/ len(metric1) +0.3 ,0,0) for a in range(len(metric1)) ]
    ax.scatter3D(metric1, metric2,metric3, c = color)
    dominating = []
    for i,_ in enumerate(metric1):
        dominated = False
        for a,_ in enumerate(metric1):
            if metric1[i] < metric1[a] and metric2[i] < metric2[a]and metric3[i] > metric3[a]:
                dominated = True
        if not dominated:
            dominating.append(i)
    dmetric1 = metric1[dominating]
    dmetric2 = metric2[dominating]
    dmetric3 = metric3[dominating]
    index = np.argsort(dmetric1)
    
    sortedmetric1 = [dmetric1[a] for a in index]
    sortedmetric2 = [dmetric2[a] for a in index]
    sortedmetric3 = [dmetric3[a] for a in index]
    
    ax.plot3D(sortedmetric1, sortedmetric2,sortedmetric3, label = "Pareto front")
    
  #  plt.plot([0.9,0.95], [0.85,0.85], label = "wished accuracy")
  #  plt.plot([0.95,0.95], [0.5,0.85], label = "wished robustness")
    plt.legend()
    plt.xlabel("robustness")
    plt.ylabel("accuracy")
    plt.zlabel("compute time")
    
    if savepath == None:
        plt.show()
    else:
        plt.savefig(savepath)


def plotpareto(metric1, metric2, metric3 = [], savepath = None):
    plt.clf()
    if not len(metric3) == 0:
        if len(metric3)< len(metric1):
            metric1 = metric1[len(metric1)-len(metric3):]
            metric2 = metric2[len(metric2)-len(metric3):]
        color = [(min(1,max(0,1- (a - np.min(metric3)) / (0.002- np.min(metric3)))),0,0) for a in metric3]
    else: 
        color= [(0.7* a/ len(metric1) +0.3 ,0,0) for a in range(len(metric1)) ]
    plt.scatter(metric1, metric2, c = color)
    dominating = []
    for i,_ in enumerate(metric1):
        dominated = False
        for a,_ in enumerate(metric1):
            if metric1[i] < metric1[a] and metric2[i] < metric2[a]:
                dominated = True
        if not dominated:
            dominating.append(i)
    dmetric1 = metric1[dominating]
    dmetric2 = metric2[dominating]
    index = np.argsort(dmetric1)
    
    sortedmetric1 = [dmetric1[a] for a in index]
    sortedmetric2 = [dmetric2[a] for a in index]
    
    plt.plot(sortedmetric1, sortedmetric2, label = "Pareto front")
    
    plt.plot([0.9,0.95], [0.85,0.85], label = "wished accuracy")
    plt.plot([0.95,0.95], [0.5,0.85], label = "wished robustness")
    plt.legend()
    plt.xlabel("robustness")
    plt.ylabel("accuracy")
    
    if savepath == None:
        plt.show()
    else:
        plt.savefig(savepath)
if __name__ == '__main__':
    plotpareto(np.genfromtxt('results/sst2small_basic/robustnesses.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2small_basic/accuracies.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2small_basic/time.csv', delimiter=',')[:-1])
    plotpareto(np.genfromtxt('results/sst2small_custom/robustnesses.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2small_custom/accuracies.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2small_custom/time.csv', delimiter=',')[:-1])
    plotpareto(np.genfromtxt('results/sst2_small_advanced/robustnesses.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2_small_advanced/accuracies.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2_small_advanced/time.csv', delimiter=',')[:-1])
    
    
    
    
    
    plotpareto(np.genfromtxt('results/sst2small_basic/robustnesses.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2small_basic/accuracies.csv', delimiter=',')[:-1])
    plotpareto(np.genfromtxt('results/sst2_small_advanced/robustnesses.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2_small_advanced/accuracies.csv', delimiter=',')[:-1])
    plotpareto(np.genfromtxt('results/sst2small_custom/robustnesses.csv', delimiter=',')[:-1],np.genfromtxt('results/sst2small_custom/accuracies.csv', delimiter=',')[:-1])
    
        
    
    w_x = 0.5
    x = np.asarray(range(0,1000,1)) / 1000.0
    y = [time_score(a,w_x) for a in x]
    
    plt.plot(x,y)
    plt.xlabel("t")
    plt.ylabel("score")
    plt.show()
