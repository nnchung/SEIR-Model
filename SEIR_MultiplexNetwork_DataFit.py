########################################################################################################################
# This code simulates simple SEIR model with social connection represented by a multiplex network.
# Input: (1) Data file: SG_COVID.csv. Real data is used for the calibration of simulation dynamics.
#        (2) Parameters: parameters for SEIR model can be modified under the section "Set parameters".
# Output: (1) Figure1.pdf plots the reported cases over time
#         (2) Figure2.pdf plots the reproduction number over time
#         (3) Simulated data is stored in a npz file

# Real-world social interaction is often too complex to be represented by ideal complex network models such as the
# Erdos-Renyi random network and the scale-free network. Here, we build a multiplex network which is composed of
# multiple overlapping networks that describe the various types of social connections between agents to study the
# dynamics of the epidemic outbreak. Specifically, our multiplex network consists of a household network, a dormitory
# network, a workplace network, a temporal crowd network, and a temporal social gathering network. Note that this list
# of networks is not exhaustive. In principle, any community which is socially connected in a significantly different
# way and is not in the existing list of communities in the model should be added as a separate layer in the
# multiplex network.

# The SG COVID data includes number of imported cases, number of infected cases within the community, number of infected
# cases among workers who stay in domitories, and number o f infected cases among those hold work permits.
# Simulation start 2 days before the starting date in the data (i.e. when the first covid case was reported)

# Reference: N. N. Chung and L. Y. Chew. Modelling Singapore COVID-19 pandemic with a SEIR multiplex network model.
# Scientific Reports 11, 10122, (2021).

########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx
import csv

###################################################### Set parameters ##################################################
N = 1000000                                            # population size
N95 = int(0.945*N)                                     # population stay in residential estate, exclude dormitories
Te = 4                                                 # mean incubation period
Ti = 2                                                 # delays from onset to isolation, i.e. not spreading the virus
scale = 0.9                                            # parameter for gamma distribution of incubation period
p = 0.1                                                # probability of infection
Nens = 25                                              # number of realizations
best_fit = 3                                           # number of realization used to fit real data
########################################### Parameters for modes of contact ############################################
# Two values are used for two different phases of the period under study (i.e. before and during lockdown)
Fc1 = 500                                              # frequency for population interacting in crowd per day
Fc2 = 200
pweak = 0.1*p                                          # infection rate is lower for weak contact in crowd
Nc_avg = 50                                            # average crowd size

Fg1 = 200                                              # frequency for population interacting in gathering per day
Fg2 = 10
Ng_avg = 50                                            # size of gathering
kg = 8                                                 # social connection, or degree of the contact network in gathering

Nh_avg = 2.5                                           # average household size
Ndorm = 1000                                           # size of overcrowded housing (we use a fixed average size)
P_employ = 0.55                                        # employment rate (including students)

plink = 0.95                                           # probability of interaction among family members, family members are more likely to interact
plink_inter = 0.2                                      # links between agents in households, in term of fraction of residential population size
plink_poor = 0.06                                      # probability of interaction among agents in crowded housing, average probability is small considering the large population in crowded housing
interlink = 4400                                       # a small number of links between agents in crowded housing
plink_fp = 0.0001                                      # links between hoseholds and crowded housings, in term of fraction of population size
plink_workplace = 0.7                                  # probability of interaction among agents in workplace
plink_inter_workplace = 0.04                           # links between agents in workplaces, in term of fraction of workplace population size
plink_family_workplace = 0.01                          # links between agents in workplaces and households, in term of fraction of workplace population size

###################################################### Load data #######################################################
reader = csv.reader(open("SG_COVID.csv"))
CB = 76 + 2                                            # the day when lockdown (i.e. circuit breaker) was implemented
seed_day = []
community = []
wpass = []
dom = []
j = 0
for row in reader:
    if j > 0:
        nodata = 0
        try:
            seed_day.append(float(row[2]))             # imported cases
        except ValueError:
            nodata = 1
        if nodata == 0:
            community.append(float(row[3]))            # infected cases within the community, excluding workers from dormitories
            wpass.append(float(row[4]))                # infected cases who hold work permit
            dom.append(float(row[5]))                  # infected cases within dormitories
    j += 1

data_c = np.array(seed_day)+np.array(community)                                    # infected cases within the community + imported cases
data_all = np.array(seed_day)+np.array(community)+np.array(wpass)+np.array(dom)    # all infected cases, by day
sgcovid_c = np.cumsum(data_c)                                                      # cumulative infected cases within the community
sgcovid_all = np.cumsum(data_all)                                                  # cumulative infected cases
T = len(seed_day) + 2

################################### Generate multiplex network layer by layer ##########################################
Cluster = np.zeros(N,dtype=int)                                                    # cluster which each agent belongs to
G = [ [] for _ in range(N)]                                                        # graph, G[i] record neighbors/contact of node i
E = [ [] for _ in range(N)]                                                        # edge type, E[i] record the corresponding edge types for G[i]

clusters_family = np.random.poisson(Nh_avg,int(N95/Nh_avg))                        # family sizes, follow poisson distribution
clusters_family = np.delete(clusters_family, np.where(clusters_family == 0))
Nfc = np.sum(clusters_family)

Npc = N - Nfc                                                                      # population stay in overcrowded housing (e.g., dormitories, slums, etc)
clusters_poor = Ndorm * np.ones(int(np.floor(Npc/float(Ndorm))))
clusters_poor[-1] = Ndorm + Npc % Ndorm
clusters_poor = clusters_poor.astype(int)

Nw = P_employ*N                                                                    # population in workplace (40% employment rate plus students interact in schools)
clusters_workplace = np.random.gamma(2.0,2.0,int(Nw/4.0))                          # workplace sizes, follow gamma distribution
clusters_workplace = clusters_workplace.astype(int)
clusters_workplace = np.delete(clusters_workplace, np.where(clusters_workplace == 0))
Nw = int(np.sum(clusters_workplace))

ID_work1 = np.random.choice(Nfc,Nw-int(0.9*Npc),replace=False)                      # agents from households who go to workplace
ID_work2 = np.random.choice(Npc,int(0.9*Npc),replace=False)                         # agents from dormitories who go to workplace
exchange = np.random.choice(int(0.9*Npc),int(0.016*Npc),replace=False)              # a small number of residential agents work with dorm workers, e.g. in construction site
exchange1 = ID_work1[exchange]                                                      # a small number of dormitory agents work with residential workers, e.g. in office
exchange2 = ID_work2[exchange]
ID_work1[exchange] = exchange2
ID_work2[exchange] = exchange1
ID_work = np.hstack((ID_work1,ID_work2))
Nedge = np.zeros(9,dtype=int)                                                       # number of edge of each edge type, edges are classified into different types, certain types exist during lockdown
Etype = ['within household', 'between households', 'within crowded housing', 'within crowded housing', 'within crowded housing', 'within crowded housing', 'within workplace', 'between workplaces', 'within workplace that offer essential service']

## household
x = 0                                                                               # index of the first agent in the household
c = 1                                                                               # index for current cluster
for ii in clusters_family:                                                          # create links within each household
    members = np.arange(x,x+ii)                                                     # index for family members in household ii
    for jj in range(ii):
        Cluster[members[jj]] = c                                                    # record cluster which the agent belongs to
        rp = np.random.random((ii,1))
        links = [members[y] for y in xrange(jj+1,ii) if rp[y, 0] < plink]
        for ll in links:
            G[members[jj]].append(ll)
            G[ll].append(members[jj])
            E[members[jj]].append(1)                                                # 1 represents intra-household contact
            E[ll].append(1)
            Nedge[1-1] += 1
    x += ii
    c += 1

for rr in range(int(plink_inter*Nfc)):                                              # links between households
    edge = np.random.choice(Nfc,2,replace=False)                                    # randomly choose 2 agents reside in residential estate
    neig = G[edge[0]]
    if not np.isin(edge[1],neig):
        G[edge[0]].append(edge[1])
        G[edge[1]].append(edge[0])
        E[edge[0]].append(2)                                                        # 2 represents inter-household contact
        E[edge[1]].append(2)
        Nedge[2-1] += 1

## poor condition, i.e. overcrowded housing
for ii in clusters_poor:                                                            # links within each housing
    members = np.arange(x,x+ii)
    for jj in range(ii):
        Cluster[members[jj]] = c
        rp = np.random.random((ii,1))
        links = [members[y] for y in xrange(jj+1,ii) if rp[y, 0] < plink_poor]
        for ll in links:
            G[members[jj]].append(ll)
            G[ll].append(members[jj])
            pp = np.random.random(1)
            if pp < 0.25:                                                           # 3, 4, 5 and 6 represent intra-dormitory contact, the contact are removed in phase during lockdown
                E[members[jj]].append(3)
                E[ll].append(3)
                Nedge[3-1] += 1
            elif 0.25 <= pp < 0.5:
                E[members[jj]].append(4)
                E[ll].append(4)
                Nedge[4-1] += 1
            elif 0.5 <= pp < 0.75:
                E[members[jj]].append(5)
                E[ll].append(5)
                Nedge[5-1] += 1
            else:
                E[members[jj]].append(6)
                E[ll].append(6)
                Nedge[6-1] += 1
    x += ii
    c += 1

for rr in range(interlink):                                                         # links between crowded housing
    edge = np.random.choice(Npc,2,replace=False) + Nfc
    neig = G[edge[0]]
    if not np.isin(edge[1],neig):
        G[edge[0]].append(edge[1])
        G[edge[1]].append(edge[0])
        E[edge[0]].append(6)                                                        # 6 also includes inter-dormitories contact, removed in the first phase during lockdown
        E[edge[1]].append(6)
        Nedge[6-1] += 1

## between households and crowded housing
for rr in range(int(plink_fp*N)):
    node1 = int(np.random.choice(Nfc,1))
    node2 = int(np.random.choice(Npc,1))
    neig = G[node1]
    if not np.isin(node2,neig):
        G[node1].append(node2)
        G[node2].append(node1)
        E[node1].append(6)                                                          # 6 also includes household-dorm contact
        E[node2].append(6)
        Nedge[6-1] += 1

## within workplace
x = 0
for ii in clusters_workplace:
    if np.random.random(1) < 0.85:                                                  # 7 represents 85% of workplace contact
        ty = 7
    else:
        ty = 9                                                                      # 9 represents contact of essential workers
    members = np.arange(x,x+ii)
    for jj in range(ii):                                                            # ii represents the size of employee in a workplace
        rp = np.random.random((ii,1))
        links = [members[y] for y in xrange(jj+1,ii) if rp[y, 0] < plink_workplace*(ii**(-0.2))]
        for ll in links:
            G[ID_work[members[jj]]].append(ID_work[ll])
            G[ID_work[ll]].append(ID_work[members[jj]])
            E[ID_work[members[jj]]].append(ty)
            E[ID_work[ll]].append(ty)
            Nedge[ty-1] += 1
    x += ii
    c += 1

## between workplace
for rr in range(int(plink_inter_workplace*Nw)):
    edge = np.random.choice(Nw,2,replace=False)
    neig = G[ID_work[edge[0]]]
    if not np.isin(ID_work[edge[1]],neig):
        G[ID_work[edge[0]]].append(ID_work[edge[1]])
        G[ID_work[edge[1]]].append(ID_work[edge[0]])
        E[ID_work[edge[0]]].append(8)
        E[ID_work[edge[1]]].append(8)
        Nedge[8-1] += 1

#### between workplace and household
for rr in range(int(plink_family_workplace*Nw)):
    edge1 = int(np.random.choice(Nw,1))                             # from workplace
    edge2 = int(np.random.choice(Nfc,1))                            # from household
    neig = G[ID_work[edge1]]
    if edge2 != ID_work[edge1]:
        if not np.isin(edge2,neig):
            G[ID_work[edge1]].append(edge2)
            G[edge2].append(ID_work[edge1])
            E[ID_work[edge1]].append(8)
            E[edge2].append(8)
            Nedge[8] += 1

print('Population size: ' + str(N))
print('Number of people stay in residential estate: ' + str(Nfc))
print('Number of people stay in overcrowded housing: ' + str(Npc))
print('Number of people in workplace: ' + str(Nw))
print('Mean workplace size: ' + str(np.mean(clusters_workplace)))
for typ in range(9):
    if 2 <= typ <= 5:
        if typ == 2:
            print('Number of edge ' + Etype[typ] + ' ' + str(np.sum(Nedge[typ:typ+4])))
    else:
        print('Number of edge ' + Etype[typ] + ' ' + str(Nedge[typ]))

####################################################### Variables to record ############################################
N_infected_time = np.zeros((Nens,T), dtype=np.int)                          # Number of infected individual over time, for each realization
N_reported_time = np.zeros((Nens,T), dtype=np.int)                          # Number of reported cases over time, for each realization, slight delay between infected and reported
N_community_time = np.zeros((Nens,T), dtype=np.int)                         # Number of infected individual in community over time, for each realization
N_active_spreader = np.zeros((Nens,T), dtype=np.int)                        # Number of active spreders over time, for each realization

for ens in range(Nens):
    print('Running realization ' + str(ens+1))

    ############################### Initialization ######################################
    Infected = np.zeros(N, dtype=np.int)                                    # countdown for infectious day, integer
    Exposed = np.zeros(N, dtype=np.int)                                     # countdown for incubation period, integer
    Recover = np.zeros(N, dtype=np.int)                                     # binary state, recovered or not
    Susceptible = np.ones(N, dtype=np.int)                                  # binary state, susceptible or not

    Ni = copy.copy(seed_day[0])                                             # initial cases come from imported cases, obtained from SG covid data
    Ni_t = np.zeros(T, dtype=np.int)                                        # number of infected agent at each time
    Ni_t[0] = copy.copy(Ni)                                                 # number of infected agent in the first day = imported cases
    Nr_t = np.zeros(T, dtype=np.int)                                        # number of infected agent reported
    Ncr_t = np.zeros(T, dtype=np.int)                                       # number of reported belong to community
    Nas = np.zeros(T, dtype=np.int)                                         # number of active spreaders

    for t in range(T-1):
        ################################# Import cases ###################################
        if t < T-3:
            ss = 0
            seeding = seed_day[t]                                           # imported cases on each day
            Ni = Ni + seed_day[t]                                           # number of infected case increases
            while ss < seeding:
                seed = np.random.choice(Nfc, 1, replace=False)              # assuming not from crowded housing
                if Susceptible[seed] == 1:
                    Iday = np.round(np.random.gamma(Ti,scale,1))            # infectious day for those become infectious
                    if Iday <= 0.0:
                        Iday = 1.0
                    Infected[seed] = Iday + 0.0
                    Susceptible[seed] = 0
                    ss = ss + 1

        ########################### Random weak contact in crowd ##########################
        if t < CB:                                                           # before lockdown
            Fc = copy.copy(Fc1)
            for wc in range(Fc):                                             # go through each crowd
                mode = 2
                Ncrowd = np.random.poisson(Nc_avg,1)                         # crowd size
                crowd = np.random.choice(N, Ncrowd, replace=False)           # randomly choose Ncrowd of agents to meet, assume all-to-all weak contact
                listIc = np.where(Infected[crowd] >= 1)[0]                   # the list of infectious agents in the crowd
                sus_ppl = np.where(Susceptible[crowd] >= 1)[0]               # the list of susceptible agents in the crowd
                if len(listIc) > 0:
                    rp = np.random.random((len(sus_ppl),1))                  # exposure probability, exposed if the infection rate is greater than a random number
                    ex = [crowd[sus_ppl[x]] for x in xrange(len(sus_ppl)) if rp[x, 0] < len(listIc)*pweak]              # agents who are exposed to virus
                    if len(ex) > 0:
                        te_ex = np.round(np.random.gamma(Te,scale,len(ex)))  # incubation period for exposed agents, follow gamma distribution
                        Exposed[ex] = copy.copy(te_ex)                       # countdown for incubation period
                        Susceptible[ex] = 0                                  # agents no longer susceptible
                        sources = crowd[listIc]                              # the agent gets the virus from who (a random agent in the crowd who is infectious)
                        if len(sources) > 1:
                            sourceR = np.random.choice(sources,len(ex),replace=True)
                        Ni = Ni + len(ex)

        else:                                                                 # during lockdown
            for wc in range(Fc1):
                if wc < Fc2:                                                  # if lockdown measures are implemented
                    mode = 2
                    Ncrowd = np.random.poisson(Nc_avg,1)
                    crowd = np.random.choice(N, Ncrowd, replace=False)
                    listIc = np.where(Infected[crowd] >= 1)[0]
                    sus_ppl = np.where(Susceptible[crowd] >= 1)[0]
                    if len(listIc) > 0:
                        rp = np.random.random((len(sus_ppl),1))
                        ex = [crowd[sus_ppl[x]] for x in xrange(len(sus_ppl)) if rp[x, 0] < len(listIc)*pweak]
                        if len(ex) > 0:
                            te_ex = np.round(np.random.gamma(Te,scale,len(ex)))
                            Exposed[ex] = copy.copy(te_ex)
                            Susceptible[ex] = 0
                            sources = crowd[listIc]
                            if len(sources) > 1:
                                sourceR = np.random.choice(sources,len(ex),replace=True)
                            Ni = Ni + len(ex)

        ########################### gathering ##################################
        if t < CB:                                                            # before lockdown
            if t < CB-14:
                Fg = copy.copy(Fg1)
            else:
                Fg = copy.copy(Fg2)                                           # gathering frequency started to decrease two weeks before lockdown

            for ss in range(Fg):
                Ng = np.random.poisson(Ng_avg,1)                              # gathering size, follow poisson distribution
                gathering = np.random.choice(Nfc, Ng, replace=False)
                Gg = nx.barabasi_albert_graph(Ng, kg)                         # contact network represented by scale-free network
                listIg = np.where(Infected[gathering] >= 1)[0]                # the list of infectious agents in the gathering
                for ii in listIg:
                    contacts = list(Gg[ii])
                    sus_ppl = np.where(Susceptible[gathering[contacts]] >= 1)[0]
                    if len(sus_ppl) > 0:
                        rp = np.random.random((len(sus_ppl),1))                # exposure probability
                        ex = [gathering[contacts[sus_ppl[x]]] for x in xrange(len(sus_ppl)) if rp[x, 0] < p]           # agents who are exposed to virus
                        if len(ex) > 0:
                            Vexp = np.round(np.random.gamma(Te,scale,len(ex)))
                            Exposed[ex] = copy.copy(Vexp)
                            Susceptible[ex] = 0
                            Ni = Ni + len(ex)

        ########################## household contact ############################
        if t < CB:
            listI = np.where(Infected >= 1)[0]
            active_spreaders = len(listI)
            active_spreaders_none = len(listI)
            for ii in listI:
                if ii < Nfc:
                    neig = list(G[ii])
                    type = list(E[ii])
                    household_contact = [neig[x] for x in xrange(len(neig)) if type[x] <= 2]
                    sus_neig = [int(household_contact[x]) for x in xrange(len(household_contact)) if Susceptible[int(household_contact[x])] == 1]
                    if len(sus_neig) > 0:
                        rp = np.random.random((len(sus_neig),1))
                        ex = [sus_neig[x] for x in xrange(len(sus_neig)) if rp[x, 0] < p]
                        if len(ex) > 0:
                            Vexp = np.round(np.random.gamma(Te,scale,len(ex)))
                            Exposed[ex] = copy.copy(Vexp)
                            Susceptible[ex] = 0
                            Ni = Ni + len(ex)
        else:
            listI = np.where(Infected >= 1)[0]
            active_spreaders = len(listI)
            for ii in listI:
                if ii < Nfc:
                    neig = list(G[ii])
                    type = list(E[ii])
                    household_contact1 = [neig[x] for x in xrange(len(neig)) if type[x] == 1]
                    household_contact2 = [neig[x] for x in xrange(len(neig)) if type[x] == 2]
                    hc2_violate = [hc for hc in household_contact2 if np.random.random(1)<0.05]      # assuming a small portion of the population does not follow the rule
                    household_contact = np.hstack((household_contact1,hc2_violate))
                    sus_neig = [int(household_contact[x]) for x in xrange(len(household_contact)) if Susceptible[int(household_contact[x])] == 1]
                    if len(sus_neig) > 0:
                        rp = np.random.random((len(sus_neig),1))
                        ex = [sus_neig[x] for x in xrange(len(sus_neig)) if rp[x, 0] < p]
                        if len(ex) > 0:
                            te_ex = np.round(np.random.gamma(Te,scale,len(ex)))
                            Exposed[ex] = copy.copy(te_ex)
                            Susceptible[ex] = 0
                            Ni = Ni + len(ex)

        ########################## workplace contact ############################
        if t < CB:
            if t % 7 < 5:                                                                               # 5 working days
                for ii in listI:
                    if np.isin(ii,ID_work):
                        neig = list(G[ii])
                        type = list(E[ii])
                        workplace_contact = [neig[x] for x in xrange(len(neig)) if type[x] >= 7]         # all workers go to workplace
                        sus_neig = [int(workplace_contact[x]) for x in xrange(len(workplace_contact)) if Susceptible[int(workplace_contact[x])] == 1]
                        if len(sus_neig) > 0:
                            rp = np.random.random((len(sus_neig),1))
                            ex = [sus_neig[x] for x in xrange(len(sus_neig)) if rp[x, 0] < p]
                            if len(ex) > 0:
                                Vexp = np.round(np.random.gamma(Te,scale,len(ex)))
                                Exposed[ex] = copy.copy(Vexp)
                                Susceptible[ex] = 0
                                Ni = Ni + len(ex)

        else:

            for ii in listI:
                if np.isin(ii,ID_work):
                    neig = list(G[ii])
                    type = list(E[ii])
                    workplace_contact = [neig[x] for x in xrange(len(neig)) if type[x] == 9]             # only essential workers continue to work after lockdown implemented
                    sus_neig = [int(workplace_contact[x]) for x in xrange(len(workplace_contact)) if Susceptible[int(workplace_contact[x])] == 1]
                    if len(sus_neig) > 0:
                        rp = np.random.random((len(sus_neig),1))
                        ex = [sus_neig[x] for x in xrange(len(sus_neig)) if rp[x, 0] < p]
                        if len(ex) > 0:
                            Vexp = np.round(np.random.gamma(Te,scale,len(ex)))
                            Exposed[ex] = copy.copy(Vexp)
                            Susceptible[ex] = 0
                            Ni = Ni + len(ex)

        ########################## dorm contact ############################
        if t < CB:
            for ii in listI:
                if ii >= Nfc:
                    neig = list(G[ii])
                    type = list(E[ii])
                    poor_contact3 = [neig[x] for x in xrange(len(neig)) if type[x] == 3]
                    poor_contact4 = [neig[x] for x in xrange(len(neig)) if type[x] == 4]
                    poor_contact5 = [neig[x] for x in xrange(len(neig)) if type[x] == 5]
                    poor_contact6 = [neig[x] for x in xrange(len(neig)) if type[x] == 6]
                    poor_contact = np.hstack((poor_contact3,poor_contact4,poor_contact5,poor_contact6))
                    sus_neig = [int(poor_contact[x]) for x in xrange(len(poor_contact)) if Susceptible[int(poor_contact[x])] == 1]
                    if len(sus_neig) > 0:
                        rp = np.random.random((len(sus_neig),1))
                        ex = [sus_neig[x] for x in xrange(len(sus_neig)) if rp[x, 0] < p]
                        if len(ex) > 0:
                            Vexp = np.round(np.random.gamma(Te,scale,len(ex)))
                            Exposed[ex] = copy.copy(Vexp)
                            Susceptible[ex] = 0
                            Ni = Ni + len(ex)

        else:
            for ii in listI:
                if ii >= Nfc:
                    neig = list(G[ii])
                    type = list(E[ii])
                    poor_contact3 = [neig[x] for x in xrange(len(neig)) if type[x] == 3]
                    poor_contact4 = [neig[x] for x in xrange(len(neig)) if type[x] == 4]
                    poor_contact5 = [neig[x] for x in xrange(len(neig)) if type[x] == 5]
                    poor_contact6 = [neig[x] for x in xrange(len(neig)) if type[x] == 6]
                    ## Rearrangement of workers housing were carried out in phase to limit the maximum number of workers housed in a room
                    if t < CB+5:
                        poor_contact = np.hstack((poor_contact3,poor_contact4,poor_contact5,poor_contact6))
                    elif t < CB+10 and t >= CB+5:
                        poor_contact = np.hstack((poor_contact3,poor_contact4,poor_contact5))
                    elif t < CB+15 and t >= CB+10:
                        poor_contact = np.hstack((poor_contact3,poor_contact4))
                    else:
                        poor_contact = copy.copy(poor_contact3)
                    sus_neig = [int(poor_contact[x]) for x in xrange(len(poor_contact)) if Susceptible[int(poor_contact[x])] == 1]
                    if len(sus_neig) > 0:
                        rp = np.random.random((len(sus_neig),1))
                        ex = [sus_neig[x] for x in xrange(len(sus_neig)) if rp[x, 0] < p]
                        if len(ex) > 0:
                            Vexp = np.round(np.random.gamma(Te,scale,len(ex)))
                            Exposed[ex] = copy.copy(Vexp)
                            Susceptible[ex] = 0
                            Ni = Ni + len(ex)

        #############################################################################
        listItoR = np.where(Infected == 1)[0]
        Recover[listItoR] = 1
        Nr_t[t+1] = Nr_t[t] + len(listItoR)
        Ncr_t[t+1] = Ncr_t[t] + len(np.where(listItoR < Nfc)[0])
        Infected[listI] = Infected[listI] - 1

        if t < CB:
            listEtoI = np.where(Exposed == 1)[0]
            if t < 30:
                Iday = np.round(np.random.gamma(Ti+1,scale,len(listEtoI)))
            else:
                Iday = np.round(np.random.gamma(Ti,scale,len(listEtoI)))
            repl = np.where(Iday==0)[0]
            Iday[repl] = 1.0
            Infected[listEtoI] = copy.copy(Iday)
        else:
            listEtoI = np.where(Exposed == 1)[0]
            Iday = np.round(np.random.gamma(Ti,scale,len(listEtoI)))
            repl = np.where(Iday==0)[0]
            Iday[repl] = 1.0
            domi = [x for x in xrange(len(listEtoI)) if listEtoI[x]>Nfc]
            Iday[domi] = np.random.poisson(Ti,len(domi))
            Infected[listEtoI] = copy.copy(Iday)

        listE = np.where(Exposed >= 1)[0]
        Exposed[listE] = Exposed[listE] - 1
        Ni_t[t+1] = copy.copy(Ni)
        Nas[t+1] = copy.copy(active_spreaders)

    N_infected_time[ens,:] = copy.copy(Ni_t)
    N_reported_time[ens,:] = copy.copy(Nr_t)
    N_community_time[ens,:] = copy.copy(Ncr_t)
    N_active_spreader[ens,:] = copy.copy(Nas)

##################################################### Fit ##############################################################
diff = np.zeros(Nens)
for ens in range(Nens):
    diff[ens] = np.sum(abs(N_reported_time[ens,2:T] - sgcovid_all))                          # sum of square error
order = np.argsort(diff)                                                                     # arrange the errors from small to large

L = int(np.round(T/4.0))
R0 = np.zeros((L,best_fit),dtype=float)
for i in range(best_fit):
    jj = 0
    active = N_active_spreader[order[i],:]
    for t in range(0,T-4,4):
        if np.sum(active[t:t+4]) > 0:
            if t+7 <= len(seed_day):
                R0[jj,i] = np.sum(active[t+4:t+8] - seed_day[t+3:t+7]) / float(np.sum(active[t:t+4]))
            else:
                R0[jj,i] = np.sum(active[t+4:t+8]) / float(np.sum(active[t:t+4]))
        jj += 1

filename = 'SEIR_fit_N' + str(N) + '_Nens' + str(Nens) + '.npz'
np.savez(filename, v1=N_infected_time, v2=N_reported_time, v3=N_community_time, v4=N_active_spreader)

fig1 = plt.figure(num=1, figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')
plt.errorbar(np.arange(0,T),np.mean(N_reported_time[order[:best_fit],:],0),np.std(N_reported_time[order[:best_fit],:],0),color='C1',label='SEIR Model (total)')
plt.errorbar(np.arange(0,T),np.mean(N_community_time[order[:best_fit],:],0),np.std(N_community_time[order[:best_fit],:],0),color='C2',label='SEIR Model (community)')
plt.plot(np.arange(2,T),sgcovid_all,'black',linewidth=3,label='Sg COVID-19 (total)')
plt.plot(np.arange(2,T), sgcovid_c, 'blue',linewidth=2,label='Sg COVID-19 (community)')
plt.xticks([11,40,71,101], ['1 Feb','1 Mar','1 Apr','1 May'])
plt.xlabel('Date')
plt.ylabel('Number of cases')
plt.legend()
plt.subplots_adjust(left=0.16,right=0.95,bottom=0.1,top=0.95)
plt.savefig('Figure1.pdf',format='pdf',dpi=600)

fig2 = plt.figure(num=2, figsize=(6, 4.5), dpi=100, facecolor='w', edgecolor='k')
plt.errorbar(np.arange(0,4*L-4,4),np.mean(R0[:-1],1),np.std(R0[:-1],1),color='C1',marker='o',fillstyle='none',label='SEIR Model')
plt.plot(np.arange(0,T-1),np.ones(T-1), color='black')
plt.xlabel('Date')
plt.ylabel('Reproduction Number')
plt.xticks([7,36,67,97], ['1 Feb','1 Mar','1 Apr','1 May'])
plt.ylim([-0.2,4.6])
plt.legend()
plt.subplots_adjust(left=0.12,right=0.95,bottom=0.1,top=0.95)
plt.savefig('Figure2.pdf',format='pdf',dpi=600)

plt.show()
