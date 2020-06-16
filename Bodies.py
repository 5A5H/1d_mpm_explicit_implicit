import numpy as np

class MPMImplicitBody1D:
    def __init__(self, ParticleVector, Material):
        # ParticleVector = [.. [X, V] ..]
        self.X = ParticleVector[:,0]
        self.V = ParticleVector[:,1]
        self.NoMP = len(self.X)
        self.Mate = Material
        self.verbose = False

    def Newmark_V(self, U, U_n, V_n, A_n, gamma, beta, dt):
        return gamma/(beta*dt) * (U-U_n) + (1.0-gamma/beta) * V_n + dt * (1.0-gamma/(2.0*beta)) * A_n
    def Newmark_A(self, U, U_n, V_n, A_n, gamma, beta, dt):
        return 1.0/(beta*dt*dt) * (U-U_n - dt*V_n - dt*dt*(1.0/2.0 - beta)*A_n)

    def SKR(self, P, N_I, Grid, dt):
        ''''''
        # Material data
        Emod = self.Mate.Emod                              # elastic modulus
        mue  = self.Mate.mue                               # damping coefficient
        rho  = self.Mate.rho                               # density

        # MaterialPoint data: read from ()_n export to ()
        x_n   = self.x_n[P]                           # Position of this MP at the beginning of this time step
        x     = self.X[P]                             # Position of this MP at the end of this time step (part of the solution)
        V_n   = self.V[P]                             # Volume of this MP at the beginning of this time step
        V     = self.V[P]                             # Volume of this MP at the end of this time step (part of the solution)
        v_n   = self.v_n[P]                           # Velocity of this MP at the beginning of this time step
        v     = self.v[P]                             # Velocity of this MP at the end of this time step (part of the solution)
        a_n   = self.a_n[P]                           # Acceleration of this MP at the beginning of this time step
        a     = self.a[P]                             # Acceleration of this MP at the end of this time step (part of the solution)
        Eps_n = self.F_n[P]                           # Deformation of this MP at the beginning of this time step
        Eps   = self.F[P]                             # Deformation of this MP at the end of this time step (part of the solution)
        b     = self.b[P]                             # body acceleration
        m     = V * rho                               # MP Mass

        # Grid data:
        XI  = Grid.GridNodeX(N_I)
        uI  = Grid.u[N_I]

        # compute shape functions
        SHP  = np.array([(x_n - XI[1])/(XI[0] - XI[1]) , 1.0 - (x_n - XI[1])/(XI[0] - XI[1])])
        dSHP = np.array([1.0/(XI[0] - XI[1]), -1.0/(XI[0] - XI[1])])

        # displacement at the MP
        u_n = 0
        u   = SHP.dot(uI)

        # deformation increment at the MP
        dEps = dSHP.dot(uI)

        # Newmark time integration at the MP
        gamma = 1.0/2.0
        beta  = 1.0/4.0
        v = self.Newmark_V(u, u_n, v_n, a_n, gamma, beta, dt)
        a = self.Newmark_A(u, u_n, v_n, a_n, gamma, beta, dt)

        # compute deformation
        Eps    = dEps + Eps_n
        Eps_dt = (Eps-Eps_n) / dt

        # compute stresses
        Sig = Emod * Eps + mue * Eps_dt * 0

        # compute rhs
        rhs = dSHP * Sig * V - SHP * b * m + SHP * a * m

        # compute lhs
        lhs = np.outer(dSHP, dSHP) * Emod * V + np.outer(dSHP, dSHP) * mue/dt * 0 * V + np.outer(SHP, SHP) * m * (1.0/(beta*dt**2))

        # export results
        Grid.R[N_I]  += rhs
        Grid.K[np.ix_(N_I, N_I)] += lhs
        self.a[P] = a
        self.v[P] = v
        self.F[P] = Eps
        self.X[P] = x_n + u
        

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