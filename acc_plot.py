import cfg
import numpy as np
import matplotlib.pyplot as plt

LABEL = 'acc'  #* loss || acc || r2

Epoch = list(range(1, cfg.NUM_EPOCHS + 1))
print(Epoch)

plt.xlabel("Epoch")


if LABEL == 'loss':
    Train_Loss = np.load(cfg.ACC_FOLDER + '/training_loss.npy', allow_pickle=True)
    Test_Loss = np.load(cfg.ACC_FOLDER + '/testing_loss.npy', allow_pickle=True)
    plt.ylabel("Loss")
    plt.plot(Epoch, Train_Loss, color='red', linestyle='-', label='Loss during Training phase')
    plt.plot(Epoch, Test_Loss, color='blue', linestyle='-', label='Loss during Testing phase')
    print("Min Train Loss: ", Train_Loss.min())
    print("Min Test Loss: ", Test_Loss.min())


if LABEL == 'acc':
    Train_Acc = np.load(cfg.ACC_FOLDER + '/training_acc.npy', allow_pickle=True)
    Test_Acc = np.load(cfg.ACC_FOLDER + '/testing_acc.npy', allow_pickle=True)
    plt.ylabel("Accuracy")
    plt.plot(Epoch, Train_Acc, color='red', linestyle='-', label='Accuracy during Training phase')
    plt.plot(Epoch, Test_Acc, color='blue', linestyle='-', label='Accuracy during Testing phase')
    print("Max Train Accuracy: ", Train_Acc.max())
    print("Max Test Accuracy: ", Test_Acc.max())


if LABEL == 'r2':
    Train_R2 = np.load(cfg.ACC_FOLDER + '/training_r2.npy', allow_pickle=True)
    Test_R2 = np.load(cfg.ACC_FOLDER + '/testing_r2.npy', allow_pickle=True)
    plt.ylabel("R2")
    plt.plot(Epoch, Train_R2, color='red', linestyle='-', label='R2 during Training phase')
    plt.plot(Epoch, Test_R2, color='blue', linestyle='-', label='R2 during Testing phase')
    print("Max Train R2: ", Train_R2.max())
    print("Max Test R2: ", Test_R2.max())


plt.legend()
plt.show()