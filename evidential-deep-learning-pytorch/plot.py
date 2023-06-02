import numpy as np
import matplotlib.pyplot as plt

import torch

def rotate_img(x, deg):
    import scipy.ndimage as nd
    return nd.rotate(x.reshape(28,28),deg,reshape=False).ravel()

def plot_util(img, clf, device = 'cpu'):
    # idx = np.where(test_labels == 9)[0][5]
    # img = test_inputs[idx]
    K = 10
    Mdeg = 180 
    Ndeg = int(Mdeg/10)+1
    ldeg = []
    lp = []
    lu=[]
    threshold = 0.5
    scores = np.zeros((1,K))
    rimgs = np.zeros((28,28*Ndeg))
    with torch.no_grad():
        for i,deg in enumerate(np.linspace(0,Mdeg, Ndeg)):
            nimg = rotate_img(img,deg)
            p_pred_t = clf.forward(torch.FloatTensor(np.expand_dims(nimg, 0)).to(device))
            p_pred_t = p_pred_t.detach().cpu().numpy()
            scores += p_pred_t >= threshold
            ldeg.append(deg)
            lp.append(p_pred_t[0])
            nimg = nimg.reshape(28,28)
            rimgs[:,i*28:(i+1)*28] = nimg

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:,labels]
    c = ['black','blue','red','brown','purple','cyan', 'orange']
    marker = ['s','^','o', 'x', '']*2
    labels = labels.tolist()
    for i in range(len(labels)):
        plt.plot(ldeg,lp[:,i],marker=marker[i],c=c[i])
    plt.legend(labels)

    plt.xlim([0,Mdeg])  
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')
    plt.show()

    plt.figure(figsize=[6.2,100])
    plt.imshow(1-rimgs,cmap='gray')
    plt.axis('off')
    plt.show()

def edl_plot_util(img, clf, device = 'cpu'):
    K = 10
    Mdeg = 180 
    Ndeg = int(Mdeg/10)+1
    ldeg = []
    lp = []
    lu=[]
    threshold = 0.5
    scores = np.zeros((1,K))
    rimgs = np.zeros((28,28*Ndeg))
    with torch.no_grad():
        for i,deg in enumerate(np.linspace(0,Mdeg, Ndeg)):
            nimg = rotate_img(img,deg)
            p_pred_t, alpha = clf.forward(torch.FloatTensor(np.expand_dims(nimg, 0)).to(device))
            p_pred_t = p_pred_t.detach().cpu().numpy()
            alpha = alpha.detach().cpu().numpy()
            u = K/np.sum(alpha, axis = -1)
            lu.append(u.mean())
            scores += p_pred_t >= threshold
            ldeg.append(deg)
            lp.append(p_pred_t[0])
            nimg = nimg.reshape(28,28)
            rimgs[:,i*28:(i+1)*28] = nimg
    print(labels)
    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:,labels]
    c = ['black','blue','red','brown','purple','cyan']
    marker = ['s','^','o']*2
    labels = labels.tolist()
    for i in range(len(labels)):
        plt.plot(ldeg,lp[:,i],marker=marker[i],c=c[i])

    labels += ['uncertainty']
    plt.plot(ldeg,lu,marker='<',c='red')

    plt.legend(labels)

    plt.xlim([0,Mdeg])  
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')
    plt.show()

    plt.figure(figsize=[6.2,100])
    plt.imshow(1-rimgs,cmap='gray')
    plt.axis('off')
    plt.show()

    return lp