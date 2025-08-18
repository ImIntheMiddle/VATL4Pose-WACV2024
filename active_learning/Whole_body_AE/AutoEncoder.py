import torch
import torch.nn as nn
from active_learning.Whole_body_AE.hybrid_feature import compute_hybrid

class WholeBodyAE(nn.Module):
    def __init__(self, z_dim=2, kp_direct=False):
        super(WholeBodyAE, self).__init__()
        self.z_dim = z_dim
        if kp_direct: # if True, use keypoints as input of AE directly. size: 15*3 = 45
            self.input_dim = 51
        else: # if False, use hand-crafted feature as input of AE. size: 15*2 + 8 = 38
            self.input_dim = 38 # for JRDB-Pose, this dim will be 42
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 24),
            nn.ReLU(True),
            nn.Linear(24, 12),
            nn.ReLU(True),
            nn.Linear(12, 7),
            nn.ReLU(True),
            nn.Linear(7, self.z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 7),
            nn.ReLU(True),
            nn.Linear(7, 12),
            nn.ReLU(True),
            nn.Linear(12, 24),
            nn.ReLU(True),
            nn.Linear(24, self.input_dim),
            nn.Sigmoid()
        )
        print(f"WholeBodyAE: z_dim: {self.z_dim}, input_dim: {self.input_dim}")
        print("Total parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat

# test code (test of WholeBodyAE, WPU
if __name__=="__main__":
    model_hybrid = WholeBodyAE(z_dim=2, kp_direct=False)
    model_hybrid.load_state_dict(torch.load("pretrained_models/wholebodyAE/Hybrid/WholeBodyAE_zdim2.pth"))
    model_hybrid.eval()
    criterion = nn.MSELoss()

    bbox_1 = [185.73092360837924, 68.4937712344281, 231.99849292450213, 288.6260002737919]
    valid_keypoints_1 = [ # 17 keypoints, each of which has 3 values (x, y, vis)
        321.8581623550401,116.74502712477396,1.0,
        285.04906333630686,132.43523316062175,1.0,
        294.75468331846565,83.3160621761658,1.0,
        0.0,0.0,0.0,
        0.0,0.0,0.0,
        274.7725245316682,146.1139896373057,0.0,
        298.28724353256024,166.36528497409327,1.0,
        327.5744870651205,145.28238341968913,1.0,
        328.70676691729324,222.33464566929132,1.0,
        342.0,109.5,1.0,
        376.0,187.5,1.0,
        233.09545049063337,252.43523316062175,1.0,
        277.0561998215879,265.4922279792746,1.0,
        229.0,345.5,1.0,
        316.44959857270294,348.80829015544043,1.0,
        0.0,0.0,0.0,
        0.0, 0.0, 0.0]
    invalid_keypoints_1 = [300,100,1,300,130,1,300,80,1,0,0,0,0,0,0,300,150,0,300,170,1,300,150,1,300,200,1,300,130,1,
                           250,150,1,300,250,1,300,260,1,300,350,1,300,170,0,0,0,0,0,0,0] # size: 17*3 = 51

    input = compute_hybrid(bbox_1, valid_keypoints_1)
    output = model_hybrid(torch.Tensor(input))
    WPU = criterion(output, input)
    print("\nWPU_valid: ", WPU.item())

    input = compute_hybrid(bbox_1, invalid_keypoints_1)
    output = model_hybrid(input)
    WPU = criterion(output, input)
    print("WPU_invalid: ", WPU.item())

    bbox_2 = [191.77340983969316,76.64779161947905,227.3081193692446,282.35220838052095]
    valid_keypoints_2 = [331.812030075188,117.46456692913385,1.0,
292.47100802854595,134.30051813471502,1.0,303.8893844781445,85.18134715025907,1.0,0.0,0.0,0.0,0.0,0.0,0.0,302.7475468331847,144.24870466321244,0.0,297.57091882247994,
162.74352331606218,1.0,336.7163247100803,131.39119170984455,1.0,313.30075187969925,235.06692913385828,1.0,357.0,117.5,1.0,376.7163247100803,230.01295336787564,1.0,
237.6628010704728,254.9222797927461,1.0,278.76895628902764,265.4922279792746,1.0,232.0,349.5,1.0,316.44959857270294,348.80829015544043,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
    invalid_keypoints_2 = [300,100,1,300,130,1,300,80,1,0,0,0,0,0,0,300,150,0,300,170,1,300,150,1,300,200,1,300,130,1,
                            250,150,1,300,250,1,300,260,1,300,350,1,300,170,0,0,0,0,0,0,0] # size: 17*3 = 51

    input = compute_hybrid(bbox_2, valid_keypoints_2)
    output = model_hybrid(input)
    WPU = criterion(output, input)
    print("\nWPU_valid: ", WPU.item())
    input = compute_hybrid(bbox_2, invalid_keypoints_2)
    output = model_hybrid(input)
    WPU = criterion(output, input)
    print("WPU_invalid: ", WPU.item())
    print("Done!")