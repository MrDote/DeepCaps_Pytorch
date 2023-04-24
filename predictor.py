from model import DeepCapsModel
import torch
import cfg
import numpy as np



BATCH_SIZE = 2


if __name__ == "__main__":

    deepcaps = DeepCapsModel(num_class=2, img_height=64, img_width=64, device="cuda:0").to("cuda:0")
    deepcaps.load_state_dict(torch.load(cfg.CHECKPOINT_FOLDER + '/simard_epoch_200.pth'))

    X = np.load(f'../CapsNet_Anton/ColorMass/images/images_cm_grey_64.npy')
    data = torch.from_numpy(X).float()

    CapsPred = []
    I=1

    deepcaps.eval()
    for i in range(0, int(data.shape[0]/BATCH_SIZE)):
        i*=BATCH_SIZE
        datasample = data[i:I*BATCH_SIZE]
        datasamplecuda = datasample.cuda()
        _, _, _, prediction = deepcaps(datasamplecuda)
        Prednpy = np.array(prediction.to('cpu').detach())
        print(I*BATCH_SIZE - 2)

        for i in range(BATCH_SIZE):
            # pred = 0 if Prednpy[i][0] < 0.5 else 1
            pred = Prednpy[i]
            # print(pred)
            CapsPred.append(pred)
        
        I+=1
    
    np.save(f'../CapsNet_Anton/ColorMass/preds/preds_simard_deepcaps', CapsPred)