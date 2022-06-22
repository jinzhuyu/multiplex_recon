import matplotlib.pyplot as plt
import numpy as np
class plotfuncs:
    def format_fig(size_scale=1):
    
        # from matplotlib import pyplot as plt
        SMALL = 13*size_scale
        MEDIUM = 15*size_scale
        LARGE = 16*size_scale
        
        # plt.style.use('classic')
        
        plt.rcParams["font.family"] = "Arial"  #Comic Sans MS, Arial, Helvetica Neue
        plt.rcParams['font.weight']= 'normal'
        plt.rcParams['figure.figsize'] = (6, 6*3/4)
        plt.rcParams['figure.titlesize'] = LARGE   
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['figure.dpi'] = 400
    
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.axisbelow'] = True
    
        plt.rcParams['axes.titlepad'] = LARGE + 2  # title to figure
        plt.rcParams['axes.labelpad'] = 3.5 # x y labels to figure
        plt.rc('axes', titlesize=MEDIUM, labelsize=MEDIUM, linewidth=1.25)    # fontsize of the axes title, the x and y labels
        
        
        plt.rcParams['ytick.right'] = False
        plt.rcParams['xtick.top'] = False
        # plt.rcParams['xtick.minor.visible'] = True
        # plt.rcParams['ytick.minor.visible'] = True
    
        plt.rc('lines', linewidth=1.8, markersize=5) #, markeredgecolor='none')
        
        plt.rc('xtick', labelsize=MEDIUM)
        plt.rc('ytick', labelsize=MEDIUM)
        
        # plt.rcParams['xtick.major.size'] = 5
        # plt.rcParams['ytick.major.size'] = 5
    
        plt.rcParams['axes.formatter.useoffset'] = False # turn off offset
        # To turn off scientific notation, use: ax.ticklabel_format(style='plain') or
        # plt.ticklabel_format(style='plain')
        
        plt.rcParams['legend.fontsize'] = SMALL
        plt.rcParams["legend.fancybox"] = True
        plt.rcParams["legend.loc"] = "best"
        plt.rcParams["legend.framealpha"] = 0.5
        plt.rcParams["legend.numpoints"] = 1
    
        
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.dpi'] = 800
        
        
    def linestyles():
        linestyles = [('solid',               (0, ())),
                      ('dashdotted',          (0, (3, 5, 1, 5))),
                      ('dotted',              (0, (1, 5))),
                      ('densely dotted',      (0, (1, 1))),
                  
                      ('loosely dashed',      (0, (5, 10))),
                      ('dashed',              (0, (5, 5))),
                      ('densely dashed',      (0, (5, 1))),
                  
                      ('loosely dashdotted',  (0, (3, 10, 1, 10))),
                      ('loosely dotted',      (0, (1, 10))),
                      ('densely dashdotted',  (0, (3, 1, 1, 1))),
                  
                      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        return linestyles



def npprint(A, n_space=2):
     assert isinstance(A, np.ndarray), "input of npprint must be ndarray"
     if A.ndim==1 :
         print(' '*n_space, A)
     else:
         for i in range(A.shape[1]):
             npprint(A[:,i])


def sample_deg_corr(G, f=0.1, edges=None, probs=None):
    '''
    Parameters:
    ----------
    G:         network
    edges:     connections
    probs:     degree correlation related probabilities
    f:         fraction of edges to be sampled
    '''
    # prob = np.array([G.degree(u)**alpha for u in G])
    # prob = prob/prob.sum()
    if edges is None:
        edges = sorted(G.edges())

    m = G.number_of_edges()
    indices=np.random.choice(range(m),size=int(m*f),replace=False,p=probs)
    # returns a copy of the graph G with all of the edges removed
    G_observed=nx.create_empty_copy(G)
    G_observed.add_edges_from(edges[indices])
    return G_observed


def sample_random_walk(G,nodes,f=0.1):
    '''
    Parameters:
    ----------
    G:         network
    nodes:     vertices
    f:         fraction of edges to be sampled
    '''
    seed_node = np.random.choice(nodes,size=1)[0]
    G_observed=nx.create_empty_copy(G)
    m = G.number_of_edges()
    size, sample,distinct=int(m*f),[seed_node],set()

    while len(distinct) < 2*size:
        u = sample[-1]
        neighbors = list(G.neighbors(u))
        v = np.random.choice(neighbors)
        sample.append(v)
        distinct.add((u,v))
        distinct.add((v,u))
        G_observed.add_edge(u,v)
    return G_observed

           
def sample_snow_ball(G,nodes,f=0.1):
    '''
    Parameters:
    ----------
    G:         network
    nodes:     vertices
    f:         fraction of edges to be sampled
    '''
    m = G.number_of_edges()
    size = int(m*f)

    while True:
        seed_node = np.random.choice(nodes,size=1)[0]
        G_observed=nx.create_empty_copy(G)
        visited=set()

        members = [seed_node]
        sz_sampled = 0
        found = False
        while len(members)>0:
            u = members[0]
            del members[0]

            if u in visited:
                continue

            visited.add(u)

            for v in G.neighbors(u):
                if v in visited:
                    continue

                members.append(v)
                G_observed.add_edge(u,v)
                sz_sampled += 1
                if sz_sampled == size:
                    return G_observed
    return G_observed

def main():
    pass

if __name__ == '__main__':
    main() 
else:             
    plotfuncs.format_fig()
    linestyles = plotfuncs.linestyles()
