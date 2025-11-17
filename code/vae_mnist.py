import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(28*28,512), nn.ReLU(), nn.Linear(512,256), nn.ReLU())
        self.mu = nn.Linear(256,d); self.lv = nn.Linear(256,d)
        self.dec = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU(), nn.Linear(512,28*28), nn.Sigmoid())
    def encode(self,x): h=self.enc(x); return self.mu(h), self.lv(h)
    def reparam(self,mu,lv): std=(0.5*lv).exp(); return mu+std*torch.randn_like(std)
    def decode(self,z): x=self.dec(z); return x.view(-1,1,28,28)
    def forward(self,x):
        mu,lv=self.encode(x); z=self.reparam(mu,lv); xh=self.decode(z)
        recon=F.mse_loss(xh,x,reduction="mean"); kl=(-0.5*(1+lv-mu.pow(2)-lv.exp())).mean()
        return xh, recon+1e-3*kl

transform=transforms.ToTensor()
train=datasets.MNIST("./mnistdata",train=True,download=True,transform=transform)
loader=DataLoader(train,batch_size=128,shuffle=True,num_workers=4,pin_memory=True)

model=VAE().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3)
out=Path("/scratch/pkadukar/soil_proj/runs/mnist_vae"); out.mkdir(parents=True,exist_ok=True)

for ep in range(3):
    model.train(); tot=0
    for x,_ in loader:
        x=x.to(device); _,loss=model(x); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    print(f"epoch {ep+1}: loss={tot/len(loader):.3f}")

model.eval()
with torch.no_grad():
    x,_=next(iter(loader)); x=x.to(device)[:32]; xh,_=model(x)
    grid=make_grid(torch.cat([x.cpu(),xh.cpu()],0), nrow=32)
    save_image(grid, str(out/"recons.png"))
print("Saved:", out/"recons.png")
