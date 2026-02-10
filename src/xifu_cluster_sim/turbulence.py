from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy import integrate

class KolmogorovPowerSpectrum:
    """
    Kolmogorov power spectrum
    """
    
    def __init__(self, 
                 sigma = 250., 
                 inj_scale = 300.,
                 alpha = 11/3):
        self.sigma = sigma
        self.inj_scale = inj_scale
        self.diss_scale = 1e-3
        self.alpha = alpha

    def __call__(self, k):
        r"""Kolmogorov power spectrum

        $$\mathcal{P}_{3D}(k)= \sigma^2 \frac{e^{-\left(k/k_{\text{inj}}\right)^2} e^{-\left(k_{\text{dis}}/k\right)^2} k^{-\alpha} }{\int 4\pi k^2  \, e^{-\left(k/k_{\text{inj}}\right)^2} e^{-\left(k_{\text{dis}}/k\right)^2} k^{-\alpha} \text{d} k}$$
        
        """
        


        k_inj = 1/self.inj_scale
        k_dis = 1/self.diss_scale
        
        #sigma = 10**log_sigma
        
        k_int = jnp.geomspace(k_inj/20, k_dis*20, 1000)
        norm = integrate.trapezoid(
            4*jnp.pi*k_int**3*jnp.exp(-(k_inj / k_int) ** 2) * jnp.exp(-(k_int/ k_dis) ** 2) * (k_int) ** (-self.alpha), 
            x=jnp.log(k_int)
                                  )
        res = jnp.where(k > 0, 
                        jnp.exp(-(k_inj / k) ** 2) * jnp.exp(-(k/ k_dis) ** 2) * (k) ** (-self.alpha),
                        0.
                       )

        return self.sigma**2 * res / norm