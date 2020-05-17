import numpy as np

class ElasticMaterial1D:
    def __init__(self, Emod, rho):
        self.Emod = Emod
        self.rho  = rho

    def Stress(self, F):
        Eps = F - 1.0
        return self.Emod * Eps

class MPMGrid1D:
    def __init__(self, lmin, lmax, NoCells):
        self.lmin = lmin
        self.lmax = lmax
        self.dx   = (lmax-lmin)/NoCells
        self.NoCells = NoCells
        self.NoNodes = NoCells +1

    def GridNodes(self, x_P):
        # compute left and right nodal indexes
        left_node_id  = np.floor(x_P/self.dx)
        right_node_id = left_node_id + 1
        return np.array([left_node_id, right_node_id], dtype=int)

    def GridNodeX(self, NodeIDs):
        # compute left and right nodal coordinates
        return np.array([i*self.dx for i in NodeIDs])

class MPMBody1D:
    def __init__(self, ParticleVector, Material):
        # ParticleVector = [.. [X, V] ..]
        self.X = ParticleVector[:,0]
        self.V = ParticleVector[:,1]
        self.NoMP = len(self.X)
        self.Mate = Material
        self.verbose = False

    def ExplixitRHS(self, P, N_I, Grid):
        '''Returns: local mass-, velocity-, momentum-, and force- vector for particle P'''
        # localize (in explicit it always depends on the past)
        x = self.x_n[P]
        V = self.V_n[P]
        v = self.v_n[P]
        m = self.m[P]
        b = self.b[P]
        Sig = self.S_n[P]

        # get nodal coodinates from grid
        X_I = Grid.GridNodeX(N_I)
        if self.verbose : print('My Nodes X        : ',X_I)

        # compute shape functions
        SHP_I  = np.array([(x - X_I[1])/(X_I[0] - X_I[1]) , 1.0 - (x - X_I[1])/(X_I[0] - X_I[1])])
        dSHP_I = np.array([1.0/(X_I[0] - X_I[1]), -1.0/(X_I[0] - X_I[1])])
        if self.verbose : print('My Shape          : ',SHP_I)
        if self.verbose : print('My derived Shape  : ',dSHP_I)
        
        # compute nodal mass
        m_I  = SHP_I * m
        if self.verbose : print('My Nodal Mass     : ',m_I)
        
        # compute nodal velocity
        v_I  =  SHP_I * v
        if self.verbose : print('My Nodal Velocity : ',v_I)
        
        # compute nodal momentum
        mv_I = SHP_I * m * v
        if self.verbose : print('My Nodal Momentum : ',mv_I)
        
        # compute nodal forces
        f_I  = - dSHP_I * Sig * V + SHP_I * b * m
        if self.verbose : print('My Nodal Forces   : ',f_I)

        # export results
        Grid.m[N_I]  += m_I
        Grid.v[N_I]  += v_I
        Grid.mv[N_I] += mv_I
        Grid.f[N_I]  += f_I

    def USL(self, P, N_I, Grid, dt):
        '''Performs the USL on this body'''
        # localize
        x   = self.X[P]
        x_n = self.x_n[P]
        V   = self.V[P]
        v   = self.v[P]
        v_n = self.v_n[P]
        m   = self.m[P]
        b   = self.b[P]
        F   = self.F[P]
        F_n = self.F_n[P]
        
        # get nodal coodinates from grid
        X_I = Grid.GridNodeX(N_I)
        if self.verbose : print('My Nodes X        : ',X_I)

        # compute shape functions
        SHP_I  = np.array([(x - X_I[1])/(X_I[0] - X_I[1]) , 1.0 - (x - X_I[1])/(X_I[0] - X_I[1])])
        dSHP_I = np.array([1.0/(X_I[0] - X_I[1]), -1.0/(X_I[0] - X_I[1])])
        if self.verbose : print('My Shape          : ',SHP_I)
        if self.verbose : print('My derived Shape  : ',dSHP_I)

        # update particle acceleration
        a = Grid.a[N_I].dot(SHP_I)
        if self.verbose : print('My acceleration   : ', a)

        # update particle velocity
        v = v_n + a * dt
        if self.verbose : print('My velocity       : ', v)

        # update particle deformation
        L = Grid.v[N_I].dot(dSHP_I)
        F = (1.0 + L * dt) * F_n
        if self.verbose : print('My deformation    : ', F)

        # update particle position
        v_bar = Grid.v[N_I].dot(SHP_I)
        x = x_n + v_bar * dt
        if self.verbose : print('My position       : ', x)

        # update stresses
        Sig = self.Mate.Stress(F)
        if self.verbose : print('My stress         : ', Sig)

        # export results
        self.a[P] = a
        self.v[P] = v
        self.F[P] = F
        self.X[P] = x
        self.S[P] = Sig

    def StrainEnergy(self):
        U = 0
        for P in range(self.NoMP):
            Eps = self.F[P] - 1.0
            U += 1.0/2.0 * self.V[P] * self.S[P] * Eps
        return U

    def KineticEnergy(self):
        K = 0
        for P in range(self.NoMP):
            K += 1.0/2.0 * self.v[P] * self.v[P] * self.m[P]
        return K