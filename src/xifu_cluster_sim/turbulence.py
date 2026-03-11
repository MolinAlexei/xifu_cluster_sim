from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy import integrate

class KolmogorovPowerSpectrum:
    r"""
    Kolmogorov power spectrum

    $$\mathcal{P}_{3D}(k)= \sigma^2 \frac{e^{-\left(k/k_{\text{inj}}\right)^2} e^{-\left(k_{\text{dis}}/k\right)^2} k^{-\alpha} }{\int 4\pi k^2  \, e^{-\left(k/k_{\text{inj}}\right)^2} e^{-\left(k_{\text{dis}}/k\right)^2} k^{-\alpha} \text{d} k}$$
    
    """
    
    def __init__(self, 
                 sigma = 250., 
                 inj_scale = 300.,
                 dis_scale = 1e-3,
                 alpha = 11/3):
        """
        Initialize parameters of the power spectrum

        Parameters:
            sigma (float): Norm of the power spectrum
            inj_scale (float): Injection scale
            dis_scale (float): Dissipation scale
            alpha (float): Slope
        """
        self.sigma = sigma
        self.inj_scale = inj_scale
        self.dis_scale = dis_scale
        self.alpha = alpha

    def __call__(self, k, k_max):
        r"""
        Computes the Kolmogorov power spectrum over k. 
        When this power spectrum is used to represent a gaussian random field on a discrete grid, if the dissipation scale is smaller
        than the smallest scale available, some power is not represented on the grid.
        This power is :

        $$\mathcal{P}_{3D}(k)= \sigma^2 \int_{k_\mathrm{max}}^{\infty} 4\pi k^2  P(k) \text{d} k$$

        It is also computed and returned.

        Parameters:
            k (float or array-like): Wavenumber
            k_max (float): Maximum wavenumber represented on grid 

        Returns:
            p_k (float or array-like): Power spectrum
            factor (float): Integral of power spectrum from k_max to infinity.
        """
        


        k_inj = 1/self.inj_scale
        k_dis = 1/self.dis_scale
        
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

        k_int = jnp.geomspace(k_max, k_dis*20, 1000)
        factor = integrate.trapezoid(
                    4*jnp.pi*k_int**3 * jnp.exp(-(k_inj / k_int) ** 2) * jnp.exp(-(k_int/ k_dis) ** 2) * k_int**(-self.alpha) / norm, 
                    x = jnp.log(k_int) 
                          ) * self.sigma**2

        return self.sigma**2 * res / norm, factor