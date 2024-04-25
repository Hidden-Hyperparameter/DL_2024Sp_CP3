# GAN
1. `rand` to `randn` (silly! most important!!!)
2. `MUL_D` : use `acc.item()` to avoid gradient accumulation in `MUL_D`

## 2023-4-21 night
1. Change 1.5 to 0.15 for dataset minibatch discrimination

## 2023-4-22 night
1. VAE: 
    - Notice that ELBO is the MAXIMIZING objective
    - faulty loss function, variance is $\sigma^2$!
2. Flow: `Distribution.log_prob` method is very bad, its shape is very wrong!!!

## GAN: final

### Generator
```python
class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.label_size = label_size = 10
        self.latent_size = latent_size
        
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        self.full_size = self.latent_size + self.label_size
        self.first_layer_size = 2
        self.project = nn.Sequential(
            nn.Linear(self.full_size, 256 * self.first_layer_size * self.first_layer_size),
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=(5,5)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.c3 = nn.Sequential(
            nn.ConvTranspose2d(128,128,kernel_size=(4,4),stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.c4 = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=(4,4),stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.c5 = nn.Sequential(
        #     nn.ConvTranspose2d(64,32,kernel_size=(2,2),stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        self.c_final = nn.ConvTranspose2d(64,1,kernel_size=(1,1))
        self.cs = [self.c2,self.c3,self.c4]
        # raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def forward(self, z, label):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        label = torch.scatter(torch.zeros([label.shape[0],10]).to(device),1,label.unsqueeze(1),value=1).to(device)
        x = torch.cat((z,label),dim=1)
        x = self.project(x)
        x = x.reshape((-1,256,self.first_layer_size,self.first_layer_size))
        for ly in self.cs:
            x = ly(x)
        x = (torch.tanh(self.c_final(x))+1)/2
        # print(x.shape)
        x = x[:,:,1:29,1:29]
        return x
        # raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
```