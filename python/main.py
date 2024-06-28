import numpy as np
import matplotlib.pyplot as plt
import casadi as cs


class nlp:

    def __init__(self,N):
        self.t0=0
        self.tf=2
        self.Nx=2
        self.Nu=1
        self.T=self.tf-self.t0
        self.N=N
        self.dt=self.T/self.N
        self.t=np.linspace(self.t0,self.tf,self.N+1)
        self.opti=cs.Opti()
        self.X=self.opti.variable(Nx,N+1)
        self.U=self.opti.variable(Nu,N)
        
    def L(self,x,u):
        L=u**2
        return L    
    
    def xdot(self,x,u):
        xd1=x[1,:]
        xd2=x[1,:]+u
        return cs.vertcat(xd1, xd2)       
    
    def objective(self,Nx,Nu,N,dt,X,U):

        x=cs.SX.sym("x",Nx,1)
        u=cs.SX.sym("u",Nu,1)
        F=self.L(x,u)
        F=cs.Function("F",[x,u],[F])

        if self.objD=="Rie":
            F=F.map(N)
            F=F(X[:,1:N+1],U)
            obj=cs.dot(dt*cs.MX.ones(1,N),F)  
            return obj
        elif self.objD=="Trap":
            F=F.map(N)
            F1=F(X[:,1:N+1],U)
            F2=F(X[:,0:N],U)
            obj=cs.dot(dt/2*cs.MX.ones(1,N),F1+F2)  
            return obj

    def res(self,Nx,Nu,N,dt,X,U):

        x=cs.SX.sym("x",Nx,1)
        u=cs.SX.sym("u",Nu,1)
        F=self.xdot(x,u)
        F=cs.Function("F",[x,u],[F]) 
        F=F.map(N)

        X1=X[:,0:N]
        X2=X[:,1:N+1]
        diffX=X2-X1

        if self.dynD=="EF":    
            # Euler Forw
            R=diffX-dt*F(X1,U)
            r=R[:]
            return r
        elif self.dynD=="EB":    
            # Euler Back
            R=diffX-dt*F(X2,U)
            r=R[:]
            return r
        elif self.dynD=="Heun":
            # Heun
            K1=F(X1,U)
            K2=F(X1+dt*K1,U)
            R=diffX-0.5*dt*(K1+K2)
            r=R[:]
            return r
        elif self.dynD=="Trap":           
            # Trap
            R=diffX-0.5*dt*(F(X1,U)+F(X2,U))
            r=R[:]
            return r
        elif self.dynD=="RK45":
            # RK45
            K1=F(X1,U)
            K2=F(X1+dt*K1/2,U)
            K3=F(X1+dt*K2/2,U)
            K4=F(X1+dt*K3,U)
            R=diffX-dt/6*(K1+2*K2+2*K3+K4)
            r=R[:]
            return r
        elif self.dynD=="Her":
            # Hermite
            Xc=(X1+X2)/2+1/8*dt*(F(X1,U)-F(X2,U))
            Uc=U
            R=diffX-dt/6*(F(X1,U)+4*F(Xc,Uc)+F(X2,U))
            r=R[:]
            return r    


class opt_problem:
    
    def __init__(self,Nx,Nu,N):
        self.Nx=Nx
        self.Nu=Nu
        self.N=N

    def OPT(self,O,RD,RBC): 
        self.opti.minimize(O)
        self.opti.subject_to(RD==0)
        self.opti.subject_to(RBC==0)
        self.opti.solver("ipopt")
        self.sol=self.opti.solve()

    def PLOT(self,t,N):
            plt.style.context("pub_qual.mplstyle")
            plt.plot(t[0:N],(6-12*t[0:N]),t[0:N],self.sol.value(self.U))
            plt.xlabel("time [s]")
            plt.ylabel("$u$")
            plt.grid()
            plt.figure()
            plt.plot(t[0:N],np.abs((6-12*t[0:N])-self.sol.value(self.U))) 
            plt.xlabel("time [s]")
            plt.ylabel("|control input error|")
            plt.grid()
            plt.show()  
            print(np.linalg.norm(self.sol.value(self.U)-(6-12*t[0:N]),2))    
            print(np.linalg.norm(self.sol.value(self.X[0,:])-(3*t**2-2*t**3),2))

    def test(self,t,N):
            print(np.linalg.norm(self.sol.value(self.U)-(6-12*t[0:N]),2))    
            print(np.linalg.norm(self.sol.value(self.X[0,:])-(3*t**2-2*t**3),2))        



class opt_obj:

    def __init__(self,Nx,Nu,N,dt,X,U):
        self.objD="Rie"
        self.obj=self.objective(Nx,Nu,N,dt,X,U)


    

    
        
class opt_dynamics:


    def __init__(self,Nx,Nu,N,dt,X,U):
        self.dynD="EB"
        self.res=self.res(Nx,Nu,N,dt,X,U)

       
 
    
 


class opt_BC:

    def __init__(self,N,X,U):
        self.x0=cs.MX(cs.vertcat(0,0))
        #print(type(self.x0))
        self.res_BC=self.BC(self.x0,N,X,U)

    def BC(self,x0,N,X,U):
        return cs.vertcat(X[:,0]-self.x0,X[0,N]-2.69458*X[1,N]+1.15333)
          

OG=opt_grid(100)
OV=opt_problem(OG.Nx,OG.Nu,OG.N)
OO=opt_obj(OG.Nx,OG.Nu,OG.N,OG.dt,OV.X,OV.U)
OD=opt_dynamics(OG.Nx,OG.Nu,OG.N,OG.dt,OV.X,OV.U)
OBC=opt_BC(OG.N,OV.X,OV.U)
OV.OPT(OO.obj,OD.res,OBC.res_BC)
OV.test(OG.t,OG.N)
OV.PLOT(OG.t,OG.N)








