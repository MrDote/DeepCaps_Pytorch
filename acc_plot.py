import cfg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

LABEL = 'acc'  #* loss || acc || r2

Epoch = list(range(1, cfg.NUM_EPOCHS + 1))
print(Epoch)

matplotlib.rcParams.update({'font.size':16})
plt.xlabel("Epoch", fontsize=20)


if LABEL == 'loss':

    plt.ylabel("RMSE Loss", fontsize=15)

    ax = plt.gca()
    ax.set_ylim([1, 3])
    # start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(1.0, 3.2, 0.5))

    Train_Loss = np.load(cfg.ACC_FOLDER + '/training_loss.npy', allow_pickle=True)
    Test_Loss = np.load(cfg.ACC_FOLDER + '/testing_loss.npy', allow_pickle=True)
    plt.plot(Epoch, Train_Loss, color='grey', linestyle='-', label='Greyscale RMSE during Training phase')
    plt.plot(Epoch, Test_Loss, color='black', linestyle='-', label='Greyscale RMSE during Testing phase')
    print("Min Train Loss: ", Train_Loss.min())
    print("Min Test Loss: ", Test_Loss.min())


if LABEL == 'r2':
    Train_R2 = np.load(cfg.ACC_FOLDER + '/training_r2.npy', allow_pickle=True)
    Test_R2 = np.load(cfg.ACC_FOLDER + '/testing_r2.npy', allow_pickle=True)
    plt.ylabel("R2")
    plt.plot(Epoch, Train_R2, color='red', linestyle='-', label='R2 during Training phase')
    plt.plot(Epoch, Test_R2, color='blue', linestyle='-', label='R2 during Testing phase')
    print("Max Train R2: ", Train_R2.max())
    print("Max Test R2: ", Test_R2.max())


if LABEL == 'acc':
    plt.ylabel("Accuracy", fontsize=20)
    
    ax = plt.gca()
    ax.set_ylim([91.0, 100.5])
    # start, end = ax.get_ylim()
    # ax.yaxis.set_ticks(np.arange(65, 101, 5))

    Train_Acc = np.load(cfg.ACC_FOLDER + '/training_acc.npy', allow_pickle=True)
    Test_Acc = np.load(cfg.ACC_FOLDER + '/testing_acc.npy', allow_pickle=True)
    plt.plot(Epoch, Train_Acc, color='grey', linestyle='-', label='Greyscale Accuracy during Training phase', zorder=1)
    plt.plot(Epoch, Test_Acc, color='black', linestyle='-', label='Greyscale Accuracy during Testing phase')
    print("Max Train Accuracy: ", Train_Acc.max())
    print("Max Test Accuracy: ", Test_Acc.max())


plt.legend(fontsize=14, loc="lower right")
plt.show()