# Feynman Kac formalisms of the function space posterior


When running a particle filter (with the Bootstrap formalism), sources of bias at each filtering step include:

- Bias from the numercial approximation scheme that is used to simulate a skeleton from $\mathbb{X}_{x_{i-1}, \Delta t}$. (In 1D case, can be eliminated entirely using exact simulation)
- Bias from applying the map $z_i = F(x_{i-1}, x_i, x_{[t_{i-1},t_i]})$ to the skeleton? **To do: Ask Alex what the map is!**
- Bias from the approximate evaluation of the potential $G(x'_{i-1}, x'_i)=p(y_i| x'_{i-1}, x'_i, y_{i-1})$, due to only having a skeleton for $z_i$ available. In practice for most models, the observation density will only depend on $x_i$, thus no bias will be introduced.
