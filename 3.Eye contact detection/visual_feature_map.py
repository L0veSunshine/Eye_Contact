import torch
from network import NetA4, SubCNNA
import cv2
import numpy as np
import hiddenlayer as hl
import matplotlib.pyplot as plt


def preprocesee(file: str):
    img = cv2.imread(file)
    rimg = cv2.resize(img, (100, 60))
    rimg: np.ndarray = np.transpose(rimg, (2, 0, 1)) / 255 * 0.99 + 0.01
    rimg = rimg.reshape((-1, 3, 60, 100))
    return torch.tensor(rimg, dtype=torch.float32)


def tensor_to_img(t: torch.Tensor):
    ndarr = t.unsqueeze(2).detach().numpy()
    ondarr = (ndarr - 0.01) / 0.99 * 255
    return ondarr.astype(np.uint8)


# graph = hl.build_graph(model, (torch.zeros([1, 3, 60, 100]), torch.zeros([1, 3, 60, 100]), torch.zeros([1, 1, 1, 3])))
# graph.save("a.jpg", "jpg")

pi = preprocesee("p1l.jpg")
model = NetA4(SubCNNA)
model.load_state_dict(torch.load("61a4_epcoh-checkpoint.chk"))
model.eval()

subnet: SubCNNA = model.subcnn1

b1 = subnet.bn1(pi)
c1 = subnet.conn1(b1)
c2 = subnet.conn2(c1)
act1 = subnet.act(c2)
m1 = subnet.maxpool1(act1)
b2 = subnet.bn2(m1)
c3 = subnet.conn3(b2)
c4 = subnet.conn4(c3)
act2 = subnet.act(c4)
m2 = subnet.maxpool2(act2)
b3 = subnet.bn3(m2)
c5 = subnet.conn5(b3)
c6 = subnet.conn6(c5)
m3 = subnet.maxpool2(c6)
f = subnet.flatten(m3)
act3 = subnet.act(f)
den1 = subnet.fc1(act3)

# fig = plt.figure(figsize=[100, 1], dpi=200)
# arr = torch.squeeze(den1)
# arr = arr.unsqueeze(0)
# arr = arr.unsqueeze(2)
# i = arr.detach().numpy().astype(np.uint8)
# print(i)
# plt.imshow(i)
# plt.axis("off")
# plt.show()

# print(dir(model.subcnn1.conn1))
# print(dir(model))
fig = plt.figure(figsize=[10, 20], dpi=300)
fig.tight_layout()
for idx, img in enumerate(torch.squeeze(c4)):
    i = tensor_to_img(img)
    ax = fig.add_subplot(16, 4, idx + 1)
    ax.axis("off")
    ax.imshow(i)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
plt.show()
# fig1 = plt.figure(figsize=[2, 2], dpi=220)
# plt.imshow(tensor_to_img(torch.squeeze(c6)[4]))
# plt.axis("off")
# plt.show()
