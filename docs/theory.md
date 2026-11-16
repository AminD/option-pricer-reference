# Theory notes

The mathematics behind each module, in the order the project was built.

Notation: $S$ underlying price, $K$ strike, $T$ maturity, $r$ risk-free rate, $q$ dividend yield, $\sigma$ volatility, $N$ the standard normal distribution function.

## 1. The model

The underlying follows a geometric Brownian motion,

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t .$$

Applying Itô's lemma to $\ln S_t$ gives $d\ln S_t = (\mu - \tfrac{1}{2}\sigma^2)\,dt + \sigma\,dW_t$, hence the exact solution

$$S_T = S_0 \exp\!\Big(\big(\mu - \tfrac{1}{2}\sigma^2\big)T + \sigma\sqrt{T}\,Z\Big), \qquad Z \sim \mathcal{N}(0,1).$$

The $-\tfrac{1}{2}\sigma^2$ term is the Itô correction: the mean of $S_T$ grows at rate $\mu$, but its median grows at rate $\mu - \tfrac{1}{2}\sigma^2$.

## 2. Risk-neutral pricing

**One-step example.** A stock worth 100 goes to 120 or 80, the interest rate is zero, and a call has strike 100 (payoff 20 or 0). Holding half a share and borrowing 40 pays $0.5 \times 120 - 40 = 20$ or $0.5 \times 80 - 40 = 0$: the same as the call. The call must therefore cost $0.5 \times 100 - 40 = 10$, whatever the probability of an up move. The same price is obtained as an expectation under the probability $q^\ast = 0.5$ that makes the stock itself fairly priced: $0.5 \times 20 = 10$.

**Continuous time.** Hold one option and sell $\Delta = \partial V/\partial S$ shares. By Itô's lemma the portfolio $\Pi = V - \Delta S$ changes by

$$d\Pi = \Big(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\Big)dt - q\,\Delta S\,dt ,$$

with no $dW$ term: it is riskless, so it must earn the risk-free rate, $d\Pi = r\Pi\,dt$. This gives the **Black-Scholes PDE**

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q)S\frac{\partial V}{\partial S} - rV = 0 .$$

The drift $\mu$ has disappeared. By the Feynman-Kac theorem the solution is a discounted expectation in which the stock drifts at $r - q$ instead of $\mu$:

$$V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\big[\text{payoff}(S_T)\big].$$

Every method in this project computes this same quantity: analytically, by simulation, on a tree, or by solving the PDE.

## 3. Black-Scholes formula

For a call, $\mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]$ is a Gaussian integral that evaluates to

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad d_{1,2} = \frac{\ln(S_0/K) + (r - q \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},$$

and $P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$ for a put. $N(d_2)$ is the risk-neutral probability of finishing in the money.

**Put-call parity.** Long a call and short a put pays $S_T - K$ in all cases, so

$$C - P = S_0 e^{-qT} - K e^{-rT},$$

independently of the model. It is the first test of the implementation.

**Greeks** (call, $\varphi$ the normal density):

| Greek | Formula |
|---|---|
| Delta | $e^{-qT} N(d_1)$ |
| Gamma | $e^{-qT}\varphi(d_1) / (S_0\sigma\sqrt{T})$ |
| Vega | $S_0 e^{-qT}\varphi(d_1)\sqrt{T}$ |
| Theta | $-\dfrac{S_0 e^{-qT}\varphi(d_1)\sigma}{2\sqrt{T}} + qS_0e^{-qT}N(d_1) - rKe^{-rT}N(d_2)$ |
| Rho | $KTe^{-rT}N(d_2)$ |

## 4. Monte Carlo

With $N$ independent draws $Z_i$, the estimator $\hat V = \frac{1}{N}\sum_i e^{-rT}\,\text{payoff}(S_T^{(i)})$ is unbiased and, by the central limit theorem, has standard error $s/\sqrt{N}$ where $s$ is the sample standard deviation of the discounted payoffs.

**Antithetic variates.** Average the payoffs obtained with $Z$ and $-Z$. The variance of the pair average is $\tfrac{1}{2}\text{Var}(X)(1+\rho)$ with $\rho < 0$ for a monotone payoff.

**Control variate.** For a variable $Y$ with known mean, $X - \beta(Y - \mathbb{E}[Y])$ has the same mean as $X$ and, with $\beta = \text{Cov}(X,Y)/\text{Var}(Y)$, variance $(1 - \rho_{XY}^2)\,\text{Var}(X)$. Here $Y = e^{-rT}S_T$ for vanilla options and the geometric Asian payoff for Asian options.

**Greeks.** Pathwise: differentiate the payoff along the path, using $\partial S_T/\partial S_0 = S_T/S_0$. Likelihood ratio: differentiate the density instead, which gives the weight $Z/(S_0\sigma\sqrt{T})$ for delta and works for discontinuous payoffs.

## 5. Binomial tree

With $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ and $p = \dfrac{e^{(r-q)\Delta t} - d}{u - d}$, the tree matches the mean and variance of the GBM over each step, and

$$V = e^{-r\Delta t}\big[p\,V_u + (1-p)\,V_d\big]$$

is the one-step replication argument of section 2. For an American option, $V = \max(\text{payoff},\ \text{continuation})$ at every node. The error is $O(1/n)$ and oscillates with the position of the strike relative to the nodes.

## 6. Finite differences

**Heat equation.** With $x = \ln S$, $\tau = T - t$ and $V = e^{\alpha x + \beta\tau}u$ for suitable constants $\alpha, \beta$, the Black-Scholes PDE becomes

$$\frac{\partial u}{\partial \tau} = \frac{\sigma^2}{2}\frac{\partial^2 u}{\partial x^2}.$$

**Discretisation.** On a grid $S_i = S_{\min} + i\,\Delta S$ with central differences, the spatial operator is tridiagonal, $(LV)_i = a_i V_{i-1} + b_i V_i + c_i V_{i+1}$ with

$$a_i = \frac{\sigma^2 S_i^2}{2\Delta S^2} - \frac{(r-q)S_i}{2\Delta S},\qquad b_i = -\frac{\sigma^2 S_i^2}{\Delta S^2} - r,\qquad c_i = \frac{\sigma^2 S_i^2}{2\Delta S^2} + \frac{(r-q)S_i}{2\Delta S}.$$

The $\theta$-scheme advances in time by

$$(I - \theta\,\Delta\tau\,L)\,V^{n+1} = (I + (1-\theta)\,\Delta\tau\,L)\,V^{n},$$

with $\theta = 0$ (explicit), $1$ (implicit) or $\tfrac{1}{2}$ (Crank-Nicolson).

**Stability.** The explicit scheme requires $\Delta\tau \lesssim \Delta S^2/(\sigma^2 S_{\max}^2)$, the analogue of $\Delta t \le \Delta x^2/(2D)$. The other two are unconditionally stable. Crank-Nicolson is $O(\Delta\tau^2)$ but does not damp the high frequencies excited by the payoff kink; two implicit half-steps at the start (Rannacher) restore smooth Greeks.

**Boundary conditions.** Call: $V(0) = 0$ and $V(S_{\max}) = S_{\max}e^{-q\tau} - Ke^{-r\tau}$. Put: $V(0) = Ke^{-r\tau}$ and $V(S_{\max}) = 0$. Knock-out barrier: $V = 0$ at the barrier, which becomes the edge of the grid.

## 7. Exotic options

**Geometric Asian.** With fixings at $t_i = iT/n$, $\ln G = \frac{1}{n}\sum_i \ln S_{t_i}$ is Gaussian with

$$\mathbb{E}[\ln G] = \ln S_0 + \big(r - q - \tfrac{1}{2}\sigma^2\big)T\,\frac{n+1}{2n}, \qquad \text{Var}[\ln G] = \sigma^2 T\,\frac{(n+1)(2n+1)}{6n^2},$$

which gives a Black-Scholes-type formula.

**Barrier options.** The reflection principle gives closed forms for continuous monitoring (Reiner-Rubinstein), built from the vanilla formula and an image term weighted by $(H/S)^{2\mu}$, $\mu = (r - q - \tfrac{1}{2}\sigma^2)/\sigma^2$. In a simulation, conditionally on the end points of a step, the log-price is a Brownian bridge that crosses the barrier $H$ with probability

$$\exp\!\Big(-\frac{2\,\ln(S_i/H)\,\ln(S_{i+1}/H)}{\sigma^2\,\Delta t}\Big).$$

**Longstaff-Schwartz.** At each exercise date, going backwards, regress the discounted future cash flows of the in-the-money paths on basis functions of $S$ to estimate the continuation value $\mathbb{E}[V_{t+\Delta t}e^{-r\Delta t} \mid S_t]$, and exercise where the payoff exceeds it. Cash flows, not fitted values, are carried backwards, which limits the bias.
