## Marginal Likelihood of Dirichlet-Multinomial Model

- $\alpha_i, \ i\in [1, K]$ - Strength parameter of the Dirichlet distribution.
- $p_i, \ i \in [1, K]$ - Class probabilities
- $\Theta$ - parameters of the model
- $y_i$ - one-hot encoded vector with $y_{ij} = 1$

We treat $Mult(y_i|p_i)$ with prior of $p_i$ being $Dir(p_i|\Theta)$. $\therefore$ the negated logarithm of the marginal likelihood is given by - 

$$
\begin{align*}
\mathcal{L}_{i}(\Theta) &= -log(\int \Pi_{j=1}^Kp_{ij}^{y_{ij}} \frac{1}{B(\alpha_i)}\Pi_{j=1}^K p_{ij}^{\alpha_{ij} - 1}d\bf{p_i}) \\
&= -log(\frac{1}{B(\alpha_i)}\int \Pi_{j=1}^{K}p^{y_{ij} + \alpha_{ij} - 1} d\bf{p}) \\
&= -log(\frac{B(\alpha_i + y_i)}{B(\alpha_i)}) \\
&= -log(\frac{\Sigma_{k\neq j}\Gamma(\alpha_{ik})\Gamma(\alpha_{ij} + 1)}{\Gamma(\Sigma (\alpha_{ik}) + 1)}\frac{\Gamma(\Sigma\alpha_{ik})}{\Sigma \Gamma(\alpha_{ik})}) \\
&= -log(\frac{\Gamma(\alpha_{ij} + 1)}{\Gamma(\Sigma (\alpha_{ik}) + 1)}\frac{\Gamma(\Sigma\alpha_{ik})}{\Gamma(\alpha_{ij})}) \\ 
&= -log(\frac{\alpha_{ij}}{\Sigma\alpha_{ik}})\\
&= \Sigma_{j=1}^K y_{ij}(log(\Sigma(\alpha_{ik})) - \alpha_{ij})
\end{align*}
$$

We maximize this w.r.t $\alpha_i$. This is called Type II Maximum Likelihood.

