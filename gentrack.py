import torch
import matplotlib.pyplot as plt
from Net import Net

model = Net()

model.load_state_dict(torch.load('model.pt'))

model.eval()

log_L=[]
log_Teff=[]

track = []
age = 2952141953419
for i in range(0, age, 100000000):
    data = torch.Tensor([1, i])
    output = model(data).tolist()
    print(output)
    L = output[1]
    T = output[2]
    # if (0 > L > 10) or (0 > T > 10):
    #     print("Skip")
    #     continue
    track.append(output)

    log_L.append(L)
    log_Teff.append(T)
    print(i/age*100, '%')

plt.plot(log_L, log_Teff)
plt.xlabel('log_Teff')
plt.ylabel('log_L')
plt.gca().invert_xaxis()
plt.show()