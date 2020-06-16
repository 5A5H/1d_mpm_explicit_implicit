import numpy as np

class ElasticMaterial1D:
    def __init__(self, Emod, mue, rho):
        self.Emod = Emod
        self.mue  = mue
        self.rho  = rho

    def Stress(self, Eps, Eps_dot):
        Sig = self.Emod * Eps + self.mue * Eps_dot
        return Sig