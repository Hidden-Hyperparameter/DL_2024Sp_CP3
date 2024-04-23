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
