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
        self.c_final = nn.ConvTranspose2d(64,1,kernel_size=(1,1))
        self.cs = [self.c2,self.c3,self.c4]
        

    def forward(self, z, label):
        label = torch.scatter(torch.zeros([label.shape[0],10]).to(device),1,label.unsqueeze(1),value=1).to(device)
        x = torch.cat((z,label),dim=1)
        x = self.project(x)
        x = x.reshape((-1,256,self.first_layer_size,self.first_layer_size))
        for ly in self.cs:
            x = ly(x)
        x = (torch.tanh(self.c_final(x))+1)/2
        x = x[:,:,1:29,1:29]
        return x
```
1. Generator can't have `c5`, otherwise the gradient will disappear. 
2. The last convolution should have `kernel_size=1`, otherwise it will contain some **patterns**(e.g. 马赛克)
3. The convolution kernel size shouldn't be too large, otherwise the numbers will be too "thick", like written by a brush.
4. Increasing the convolution channel number may lead to model representation too much, in which cases the model will learn some strange patterns to fool the discriminator. On the other hand, if the channel number is too small, the model will not be able to learn much number patterns, so the variance will be small.

### Discriminator
```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_size = label_size = 10
        self.c2 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=(2,2)),
            # nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(32,2,kernel_size=(2,2),stride=2),
            nn.BatchNorm2d(2),
            # nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2)
        )
        # self.c4 = nn.Sequential(
        #     nn.Conv2d(4,4,kernel_size=(3,3),stride=2),
        #     nn.BatchNorm2d(4),
        #     # nn.Dropout2d(0.2),
        #     nn.LeakyReLU(0.2)
        # )
        self.cs = [self.c2,self.c3]
        self.out_conv_size = 338
        self.mini_batch = 16
        self.B = 100
        self.C = 4
        self.tensor = nn.Linear(self.out_conv_size,self.B * self.C)
        self.linear_size = self.out_conv_size + self.B
        self.project = nn.Sequential(
            nn.Linear(self.linear_size,self.label_size+1),
            # nn.LeakyReLU(0.2),
            # nn.Linear(64,self.label_size+1)
        )

    def forward(self, img):
        x = img
        batch_size = img.shape[0]
        x = x.reshape((-1,1,28,28))
        for ly in self.cs:
            x = ly(x)
        x = x.reshape([batch_size,-1])

        # minibatch discrimination
        o = torch.zeros([batch_size,self.B]).to(device)
        M = self.tensor(x).reshape([batch_size,1,self.B,self.C])
        for down in range(0,batch_size,self.mini_batch):
            M_mini = M[down:down+self.mini_batch]
            delta = M_mini - M_mini.transpose(0,1) # M_mini: i,1,B,C - 1,j,B,C
            delta = torch.permute(delta,[2,0,1,3]) # self.B,mini_batch,mini_batch,self.C
            delta = torch.sum(torch.abs(delta),dim=3)
            c = torch.exp(-delta)
            o[down:down+self.mini_batch] = c.sum(dim=2).T

        features = x.detach()
        final = self.project(torch.cat((x,o),dim=1))
        return final,features
```
1. `Dropout`: its effect is not clearly examined.
2. Changing the parameter in `LeakyReLU` won't help much based on my experience.
3. **Feature matching (have some small effects)**: return features in the end.
4. The discriminator can't have `c4`, otherwise it will be too strong. Similarly, the `linear` can only have one layer.
5. **Minibatch Discrimination**: a very strong technique of ensuring the variance (and diversity) of the generated images, see the paper [improvedTech](./docs/GANTrainTeqs.pdf). The implement is also very smart, which is done by LYY. (The parallelization of the computation, i.e. using `:` instead of loops, brings about 10x speedup.) 
    1. `B` and `C` values (very important): as not very intuitive, `B` should be large (~100) and `C` should be small. That is, the length of `o` should be comparable with `x`(which is around 300 after coming out from convolution layers)
    2. No modification of the value of `o` should be permitted. Even when `o` has 2 order larger than `x`, you should not devide it by 100.
    3. The most important: minibatch discrimination must be carry out with **different labels**. I used to create a variable `gen_labels`, which is like `[0,0,..,0,1,1,..,1,...]`, and even sort the labels in the dataloader `y`, but this will make things worse.

### Training
```python
def train(n_epochs, generator, discriminator, train_loader, optimizer_g, optimizer_d, device=torch.device('cuda'), save_interval=10):
    generator.to(device)
    discriminator.to(device)
    for epoch in range(n_epochs):
        train_g_loss = train_d_loss = 0
        n_batches = 0
        pbar = tqdm(total=len(train_loader.dataset))
        for i, (x, y) in enumerate(train_loader):
            # compute loss
            n_batches += x.shape[0]
            x = x.to(device)
            y = y.to(device)

            # train discriminator
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            discriminator.train()

            MINI_BATCH_SIZE = discriminator.mini_batch
            actual_size = ((len(y)//MINI_BATCH_SIZE)*MINI_BATCH_SIZE)
            y = y[:actual_size]# since minibatch discrimination required the passed in data to be multiple of batch size
            x = x[:actual_size]
        
            results_real_image,real_features = discriminator(x)
            first = F.cross_entropy(results_real_image,y).mean() # loss on dataset
            d_real_acc = (results_real_image.argmax(dim=1)==y).to(torch.float).mean().item()
            
            z = torch.randn([y.shape[0],generator.latent_size]).to(device)
            gen_images = generator(z,y)
            output,_ = discriminator(gen_images)
            second = F.cross_entropy(output,(F.one_hot((torch.ones(output.shape[0])*10).to(torch.long),num_classes=11)*0.9).to(device)) # loss on generated images, 0.9 means one-side label smoothing
            d_fake_acc = (output.argmax(dim=1)==10).to(torch.float).mean().item()
            
            d_loss = first + second
            d_loss.backward()
            optimizer_d.step()

            # train generator
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            discriminator.eval()

            z = torch.randn([y.shape[0],generator.latent_size]).to(device)
            gen_images = generator(z,y)
            output,features = discriminator(gen_images)
            g_loss_first = F.cross_entropy(output,y)
            g_loss_second = F.mse_loss(features.mean(dim=0),real_features.mean(dim=0)) # feature matching
            
            g_loss = g_loss_first  + g_loss_second
            g_acc = (output.argmax(dim=1)==y).to(torch.float).mean().item()
            g_loss.backward()
            optimizer_g.step()


            if i%2==0:
                pbar.update(x.size(0)*2)
                pbar.set_description('Ep({})GL:({:.6f})DL({:.6f})DrA({:.2f})DfA({:.2f})GA({:.2f})'.format(
                    epoch + 1, train_g_loss / n_batches, train_d_loss / n_batches,d_real_acc*100,d_fake_acc*100,g_acc*100))

            train_g_loss += g_loss.sum().item()
            train_d_loss += d_loss.sum().item()

        pbar.close()

        if (epoch + 1) % save_interval == 0:
            os.makedirs(f'./gan/{epoch + 1}', exist_ok=True)
            save_model(f'./gan/{epoch + 1}/gan.pth', generator, optimizer_g,
                       discriminator=discriminator, optimizer_d=optimizer_d)

            # sample and save images
            label = torch.arange(10).repeat(10).to(device)
            generator.sample_images(
                label, save=True, save_dir=f"./gan/{epoch + 1}/")          
```
1. **One-side Label smoothing**: use 0.9 for positive label. This will let the discriminator give some opportunties for generator to learn in the first few epochs. 
2. **Feature matching**: (explained before)

### Overall
1. The training process is very strange. What's very important is that you shouldn't give up a model so soon.

The following image is the training curve. Notice that at the first 7~8 epochs, the discriminator have absolute advantage over the generator.

![](./gan_loss.png)

2. The training process is shown in the folder [folder](../train_gan_image_history/). Notice that the generated images are very bad at the beginning, but they will be improved afterwards.

## What I have learned
1. Never tune something by yourself--use packages. For example, don't manually (automatically) adjust learning rate in order to make the training stable. You should believe `Adam`, and let it run for 100 epochs. Maybe the results would be disappointing at first, but they will be improved. (see the images in the `train_gan_image_history` folder). Specifically, avoid this kind of code:

```python
if d_acc > 0.95:
    d_lr -= 0.0001
```

2. Similarly, never manually inference the model training. Sometimes two losses may be not in the same order (e.g. the feature matching, almost always is 0). However, since the term has theoretically supports, it will be useful at some time. 

3. **Complex things don't work. Even if they work, there are simpler reasons behind them**. For example, as mentioned, I tried to sort the dataset in order to let the minibatch discrimination only work on same label. However, after all, I found that this is not reasonable since there is only one projection tensor for the minibatch information, so it can't be trained on different labels.