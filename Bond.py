import numpy as np
from scipy import integrate
import scipy.integrate as sciIntegr

class Payoff(object):
    def __init__(self, K):
        self.K = K
    def f(self):
        pass
    
class ZCBOption(Payoff):
    def f(self, S):
        return np.maximum(S - self.K,0)

class ZCBValue(Payoff):
    def f(self, S):
        return S

class Bond(object):
    def __init__(self, theta, kappa, sigma, r0=0.):
        '''
        r0    is the current level of rates
        kappa is the speed of convergence
        theta is the long term rate
        sigma is the volatility    
        '''
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.r0    = r0
        self.dt    = 0.001
    
    def B(self, t, T):
        pass
    
    def A(self, t, T): 
        pass
    
    def Exact_zcb(self, t, T):
        pass
    
    def Euler(self, M, I, T):
        pass
    
    def Yield(self, t, T, rate):
        return -1/(T-t)*self.A(t,T) + 1/(T-t)*self.B(t,T)*rate
    
    def SpotRate(self, t, T) :
        price = self.Exact_zcb(t, T)
        time  = T - t
        return (-np.log(price)/time)
    
    def ForwardRate1(self, time):
        up   = np.max(time + self.dt,0)
        down = np.max(time - self.dt,0)        

        r        = self.SpotRate(0, time)                        
        dr       = self.SpotRate(0, up) - self.SpotRate(0, down)
        dr_dt    = dr/(2*self.dt)         
        fwd_rate = r + time * dr_dt
        return fwd_rate

    def ZCB_Forward_Integral1(self, t, T):    
        val = integrate.quad(self.ForwardRate1, t, T)[0]
        return np.exp(-val)
    
    def ForwardRate2(self, time):            
        up   = np.max(time + self.dt,0)
        down = np.max(time - self.dt,0)

        dP       = self.Exact_zcb(0, down) - self.Exact_zcb(0, up)     
        dP_dt    = dP/(2*self.dt)
        P        = self.Exact_zcb(0, time)
        fwd_rate = dP_dt/P
        return fwd_rate

    def ZCB_Forward_Integral2(self, t, T):                    
        val = integrate.quad(self.ForwardRate2, t, T)[0]
        return np.exp(-val)

    def ForwardRate3(self, time):            
        up       = np.max(time + self.dt,0)
        down     = np.max(time - self.dt,0)

        dP       = np.log(self.Exact_zcb(0, down)) - np.log(self.Exact_zcb(0, up))     
        fwd_rate = dP/(2*self.dt)
        return fwd_rate 

    def ZCB_Forward_Integral3(self, t, T):                    
        val = integrate.quad(self.ForwardRate3, t, T)[0]
        return np.exp(-val)
    
    def StochasticPrice(self, VectorRates, VectorTime):  
        # VectorRates is a two dimensional array:
        # with simulated rates in columns and timesteps rates in rows
        
        # we do not need VectorRates and VectorTime at the beginning of the simulation as it is r0        
        VectorRates = VectorRates[:-1,:]
        VectorTime  = VectorTime[:-1]
        
        No_Sim = VectorRates.shape[1]
         
        price  = np.zeros(No_Sim)
        for i in range(No_Sim):    
            Rates    = VectorRates[:,i].T
            price[i] = np.exp(-(sciIntegr.simps(Rates , VectorTime)))        
        
        RangeUp_Down   = np.sqrt(np.var(price))*1.96 / np.sqrt(No_Sim)
        Mean = np.mean(price)
                
        return Mean,  Mean + RangeUp_Down, Mean - RangeUp_Down
        
    def FutureZCB(self,M, I, T_0, T_M, Bond):  
        
        self.Euler(M, I, T_0)        
        bond = Bond    
        
        # we do not need VectorRates and VectorTime at the beginning of the simulation as it is r0        
        VectorRates = self.rates[:-1,:]
        VectorTime  = self.times[:-1]
        
        No_Sim = VectorRates.shape[1]
         
        price          = np.zeros(No_Sim)
        priceUntil_FO  = np.zeros(No_Sim)
        Payoff         = np.zeros(No_Sim)
        R0             = np.zeros(No_Sim)
        for i in range(No_Sim):    
            Rates      = VectorRates[:,i].T
            R0[i]      = Rates[-1]
            Yield      = self.Yield(T_0, T_M, Rates[-1])
            ZCBPrice   = np.exp(-Yield*(T_M-T_0)) 
            Payoff[i]  = bond.f(ZCBPrice)    
            price[i]   = np.exp(-(sciIntegr.simps(Rates , VectorTime)))*Payoff[i]      
            priceUntil_FO[i]  = np.exp(-(sciIntegr.simps(Rates , VectorTime)))
        
        RangeUp_Down   = np.sqrt(np.var(price))*1.96 / np.sqrt(No_Sim)
        Mean     = np.mean(price)
        MeanRate = np.mean(R0)
        FWDValue = np.mean(Payoff)
        MeanValueUntil_FO = np.mean(priceUntil_FO)
        return Mean,  Mean + RangeUp_Down, Mean - RangeUp_Down, FWDValue, MeanValueUntil_FO, MeanRate
    
    def ExpectedRate(self,t, T):
        pass
    
    def VarianceRate(self,t, T):
        pass

    def PlotEulerSim(self,Text, No_of_Sim = 10):        
        # We plot the first No_of_Sim simulated paths 
        
        plt.plot(self.times, self.rates[:, :No_of_Sim], lw=1.5)
        plt.xlabel('time - yrs')  
        plt.ylabel('rates level')
        plt.grid(True)
        plt.title(Text + ' - the first {}'.format(No_of_Sim) + " Simulated Paths");
        plt.show()

    def PlotEulerSim_Stats(self, Text):        
        # We plot and compare the average simulation +-2 sd of all time steps vs. what we expect from the model
        
        SimAverage  = np.mean(self.rates, 1)
        SimSD       = np.sqrt(np.var(self.rates, 1))
        AnalyAverage= np.asarray([self.ExpectedRate(0, i) for i in self.times])
        AnalySD     = np.asarray([np.sqrt(self.VarianceRate(0, i)) for i in self.times])
        
        plt.plot(self.times, SimAverage,           lw=1.5, label ='Sim Mean',linestyle=':')
        #plt.plot(self.times, SimAverage + 2*SimSD, lw=1.5, label ='Sim Mean + 2*SD',linestyle=':')
        #plt.plot(self.times, SimAverage - 2*SimSD, lw=1.5, label ='Sim Mean - 2*SD',linestyle=':')
        
        plt.plot(self.times, AnalyAverage,             lw=1.5, label ='Analy Mean',linestyle='-')
        #plt.plot(self.times, AnalyAverage + 2*AnalySD, lw=1.5, label ='Analy Mean + 2*SD',linestyle='-.')
        #plt.plot(self.times, AnalyAverage - 2*AnalySD, lw=1.5, label ='Analy Mean - 2*SD',linestyle='-.')
                
        plt.legend()
        
        plt.xlabel('time - yrs')  
        plt.ylabel('rates')
        plt.grid(True)        
        plt.title(Text);        
        plt.show()