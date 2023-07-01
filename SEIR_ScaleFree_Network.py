########################################################################################################################
# This code simulates simple SEIR model with social connection represented by a scale-free network.
# Parameters for the model (e.g., population size, infection rate, etc) can be modified in the "Set parameters" section.
# Output: The code plots a figure that shows the susceptible, infected and recovered population over time.
# Reference: N. N. Chung and L. Y. Chew. Modelling Singapore COVID-19 pandemic with a SEIR multiplex network model.
# Scientific Reports 11, 10122, (2021).

########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import copy

###################################################### Set parameters ##################################################
N = 1000                                              # population size
Te = 4                                                # mean incubation period
Ti = 2                                                # delays from onset to isolation, i.e. not spreading the virus
scale = 0.9                                           # parameter for gamma distribution of incubation period
p = 0.1                                               # probability of infection
k = 8                                                 # social connection, i.e. degree of contact network
T = 100                                               # number of days simulated
Nens = 100                                            # number of realizations simulated
Nseed = 5                                             # number of initial infection

R0 = k*p*Ti                                           # simple estimation of R0 based on social contact, infection rate and infection period
print('Reproduction Number: ' + str(R0))

################################################# Generate network #####################################################
G = [ [] for _ in range(N)]                            # Graph, G[i] record neighbors of node i
degree = np.zeros(N,dtype=int)                         # degree, degree[i] record degree of node i

Nseed_BA = 4                                           # seed size to generate BA network
for jj in range(Nseed_BA):                             # full seed, i.e., all connect to all, make sure the seed forms a connected cluster
    G[jj] = [x for x in range(Nseed_BA) if x != jj]
    degree[jj] = len(G[jj])

mlink = k/2                                            # number of link to be added to each new node
pos = Nseed_BA                                         # index of new node
sumdeg = np.sum(degree[:Nseed_BA])
while pos < N:                                         # add a new node to the network
    linkage = 0
    while linkage < mlink:                             # until "mlink" number of link is added
        nodelist = np.random.choice(pos-1,1)           # choose an existing node randomly
        node = copy.copy(nodelist[0])
        if not np.in1d(pos,G[node]):                   # the two are not linked before
            if node != pos and degree[node]/(sumdeg+0.0) >= np.random.random(1):          # determine whether the node is selected to draw a link by comparing its "degree/total degree" with a random number
                G[pos].append(node)                    # add "node" to the list of neighbors of "pos"
                G[node].append(pos)
                degree[pos] += 1
                degree[node] += 1
                linkage += 1
                sumdeg += 1
    pos += 1

for ens in range(Nens):
    if ens % 10 == 0:
        print('ens: ' + str(ens))
    ##################################################### Initialization ###############################################
    Infected = np.zeros(N, dtype=np.int)                   # countdown for infectious day, integer
    Exposed = np.zeros(N, dtype=np.int)                    # countdown for incubation period, integer
    Recover = np.zeros(N, dtype=np.int)                    # binary state, recovered or not
    Susceptible = np.ones(N, dtype=np.int)                 # binary state, susceptible or not
    seed = np.random.choice(N, Nseed, replace=False)       # initial individual who are infected
    Iday = np.round(np.random.gamma(Ti,scale,len(seed)))   # infectious days for the seed, follows gamma distribution
    repl = np.where(Iday <= 0)[0]                          # if there are zero day or less, replace with 1 day
    Iday[repl] = 1.0
    Infected[seed] = copy.copy(Iday)
    Susceptible[seed] = 0                                  # seed no longer susceptible

    Ni = copy.copy(Nseed)
    Ni_t = np.zeros(T+1, dtype=np.int)                     # number of infected individual over time
    Ni_t[0] = copy.copy(Ni)
    Nr = 0
    Nr_t = np.zeros(T+1, dtype=np.int)                     # number of recovered individual over time
    Ni_t[0] = copy.copy(Nr)
    Ns_t = np.zeros(T+1, dtype=np.int)                     # number of susceptible individual over time
    Ns_t[0] = copy.copy(N)

    ############################################### Simulate the epidemics #############################################
    for t in range(T):
        listI = np.where(Infected >= 1)[0]                                                    # the list of infectious agents
        for ii in listI:                                                                      # go through the list of infectious agent
            neig = list(G[ii])                                                                # contact of infectious agent
            sus_neig = [neig[x] for x in xrange(degree[ii]) if Susceptible[neig[x]] == 1]     # contacts who are susceptible
            if len(sus_neig) > 0:
                rp = np.random.random((len(sus_neig),1))                                      # exposure probability, compared to random numbers, exposed if the infection rate is greater than a random number
                ex = [sus_neig[x] for x in xrange(len(sus_neig)) if rp[x, 0] < p]             # agents who are exposed to virus
                if len(ex) > 0:
                    te_ex = np.round(np.random.gamma(Te,scale,len(ex)))                       # incubation period for exposed agents, follow gamma distribution
                    Exposed[ex] = copy.copy(te_ex)
                    Susceptible[ex] = 0                                                       # exposed agents are no longer susceptible
                    Ni = Ni + len(ex)                                                         # update the number of infected individual

        listItoR = np.where(Infected == 1)[0]                                                 # infectious agent become recovered
        Recover[listItoR] = 1
        Nr = Nr + len(listItoR)
        Infected[listI] = Infected[listI] - 1                                                 # infectious period reduced by 1
        listEtoI = np.where(Exposed == 1)[0]                                                  # exposed agents become infectious
        Iday = np.round(np.random.gamma(Ti,scale,len(listEtoI)))                              # infectious day for those become infectious
        repl = np.where(Iday <= 0)[0]
        Iday[repl] = 1.0                                                                      # if there are zero day or less, replace with 1 day
        Infected[listEtoI] = copy.copy(Iday)
        listE = np.where(Exposed >= 1)[0]
        Exposed[listE] = Exposed[listE] - 1                                                   # incubation period reduced by 1
        Ni_t[t+1] = copy.copy(Ni)                                                             # record number of infected agent at time t
        Nr_t[t+1] = copy.copy(Nr)
        sus_agents = [x for x in xrange(N) if Susceptible[x] == 1]
        Ns_t[t+1] = len(sus_agents)

fig = plt.figure(num=1, figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')
plt.plot(np.arange(0,T+1), Ni_t/(N+0.0), 'r', label='Infected')
plt.plot(np.arange(0,T+1), Nr_t/(N+0.0), 'g', label='Recovered')
plt.plot(np.arange(0,T+1), Ns_t/(N+0.0), 'b', label='Susceptible')
plt.xlabel('Day')
plt.ylabel('Fraction of population')
plt.legend()
plt.savefig('Figure1.pdf',format='pdf',dpi=600)
plt.show()
