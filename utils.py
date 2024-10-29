# Preprocessing for proper data form

from scipy.io import loadmat
import scipy.stats
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.stats import t
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import inspect
import random
import seaborn as sns
from scipy.stats import mannwhitneyu
import pandas as pd

def groupPearR(x):
    CorrMat=[]
    PMat=[]
    for i in range (len(x[:,0])-1):
        for j in range (i+1,len(x[:,0])):
            CorrMat.append(np.corrcoef(x[i,:], x[j,:])[0, 1])
            PMat.append(pearsonr(x[i,:], x[j,:])[1])
    return CorrMat, PMat


def AddWeightededge(x, y):
    # x: group of spike data (time series x neuron #)
    # y: group of each cells' location (neuron # x (x location, y location from NMDA))
    W=nx.Graph()
    Corr, P=groupPearR(x)
    Num=0
    Totallength=int(len(x[:,0]))
    W.add_nodes_from(list(range(1,Totallength+1)))
    for i in range (len(x[:,0])):
        for j in range (i+1,len(x[:,0]+1)):
            if math.dist(y[i,0:2], y[j,0:2])!=0:
                CorrDevDis=np.divide(Corr[Num],math.dist(y[i,0:2], y[j,0:2]))
            if np.isnan(Corr[Num])==True or P[Num]>0.05 or Corr[Num]<0:
                Num+=1
                continue
            W.add_edge(i+1,j+1, weight=CorrDevDis)
            Num+=1
    return W 


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def cross_corr_Tshift(x, y):
    
    # Normalization process to make np.correlate in range -1~1
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    
    # 표준 편차가 0인 경우 처리
    if x_std == 0 or y_std == 0:
        
        return None, None, None
    
    else:
    
        x_norm = (x - x_mean) / (x_std * len(x))
        y_norm = (y - y_mean) / y_std
        corr = np.correlate(x_norm, y_norm, mode='same')
        peaks, _ = find_peaks(corr)
        if len(peaks) == 0:
            peak_idx = None
            p_value=None
            peak_Val=None
        else:
            peak_idx = 0.1*(peaks[np.argmax(corr[peaks])]-len(corr)/2)*2/len(x) # Normalize to be -1~1
            p_value=calculate_p_value(corr[int(peaks[np.argmax(corr[peaks])])], x.size)
            peak_Val=corr[int(peaks[np.argmax(corr[peaks])])]

        return peak_idx, p_value, peak_Val


def calculate_p_value(correlation, sample_size):
    
    # 예외 처리: correlation이 1 또는 -1인 경우
    if correlation == 1 or correlation == -1:
        
        return 0.0
    
    elif correlation == None:
        
        return None
    
    else:
        
        # degree of freedom
        df = int(sample_size - 2)
        
#         print(df)
#         print(correlation)
        
        # t-value
        t_value = correlation* np.sqrt(float( df / (1 - correlation**2)))

        # p-value
        p_value = 2 * (1 - t.cdf(np.abs(t_value), df))
    
        return p_value


def fwhm(x, y):
    peaks, _ = find_peaks(y)
    if len(peaks) == 0:
        return None
    peak_idx = peaks[np.argmax(y[peaks])]
    hm = (np.max(y) - np.min(y)) / 2
    left_idx = np.argmin(np.abs(y[:peak_idx] - hm))
    right_idx = peak_idx + np.argmin(np.abs(y[peak_idx:] - hm))
    return x[right_idx] - x[left_idx]


def crosscorr_FWHM(x, y, t):
    crosscorrfwhm=fwhm(t, cross_corr(x, y))
    
    return crosscorrfwhm


def AddCrCorredge(x):
    G = nx.DiGraph()
    Num=0
    Totallength=int(len(x[:,0]))
    G.add_nodes_from(list(range(1,Totallength+1)))
    
    for i in range (len(x[:,0])): # neuron#
        for j in range (i+1,len(x[:,0]+1)): 
            #crosscorrfwhm=crosscorr_FWHM(x[i,:], x[j,:],np.linspace(-len(x[0,:]),len(x[0,:]),2*len(x[0,:])+1))
            #,math.dist(y[i,0:2], y[j,0:2])
            timeshift, p, peak_Val= cross_corr_Tshift(x[i,:], x[j,:])
            if timeshift==None or p==None or p>0.05:
                continue
            elif timeshift>=0:
                G.add_edge(i+1,j+1, weight=np.exp(abs(peak_Val))/np.exp(abs(timeshift)), distance=np.exp(abs(timeshift))/np.exp(abs(peak_Val)))
            else:
                G.add_edge(j+1,i+1, weight=np.exp(abs(peak_Val))/np.exp(abs(timeshift)), distance=np.exp(abs(timeshift))/np.exp(abs(peak_Val)))
            
            Num+=1
            
    return G 

def compute_cross_correlation_relations(Num_st_neuron, Cum_Num_st, ExCSD_STs_noCt, ExLocal_STs_noCt, InCSD_STs_noCt, InLocal_STs_noCt):
    ex_csd_CrossTauPlot = [[], []]
    ex_local_CrossTauPlot = [[], []]
    in_csd_CrossTauPlot = [[], []]
    in_local_CrossTauPlot = [[], []]
    
    def process_data(x, cross_tau_plot):
        for k in range(len(x[:, 0])):  # neuron#
            for l in range(k + 1, len(x[:, 0])+1):
                timeshift, p, peaks = cross_corr_Tshift(x[k, :], x[l, :])
                if timeshift is None or p is None or p > 0.05:
                    continue
                else:
                    cross_tau_plot[0].append(timeshift)
                    cross_tau_plot[1].append(peaks)
    
    for i in range(0, len(Num_st_neuron)):
        for j in range(1, len(Num_st_neuron[i])+1):
            if i == 0:
                x = ExCSD_STs_noCt[Cum_Num_st[i][j - 1]:Cum_Num_st[i][j], :]
                process_data(x, ex_csd_CrossTauPlot)
            elif i == 1:
                x = ExLocal_STs_noCt[Cum_Num_st[i][j - 1]:Cum_Num_st[i][j], :]
                process_data(x, ex_local_CrossTauPlot)
            elif i == 2:
                x = InCSD_STs_noCt[Cum_Num_st[i][j - 1]:Cum_Num_st[i][j], :]
                process_data(x, in_csd_CrossTauPlot)
            elif i == 3:
                x = InLocal_STs_noCt[Cum_Num_st[i][j - 1]:Cum_Num_st[i][j], :]
                process_data(x, in_local_CrossTauPlot)
            else:
                break

    return ex_csd_CrossTauPlot, ex_local_CrossTauPlot, in_csd_CrossTauPlot, in_local_CrossTauPlot



def plot_weight_distribution(graphs, row_index, title, bins=20):
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.get_cmap('tab10', len(graphs))  # Using a colormap for distinct colors
    
    for graph_index, graph in enumerate(graphs):
        if isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
            weights = [data['weight'] for u, v, data in graph.edges(data=True)]
            max_weight = max(weights) if weights else 1
            normalized_weights = [weight / max_weight for weight in weights]
            
            # Create bins and calculate frequency of weights within each bin
            bin_edges = np.linspace(0, 3, bins + 1)
            hist, _ = np.histogram(weights, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            color = colors(graph_index)
            
            plt.plot(bin_centers, hist, marker='o', label=f'Graph {graph_index+1}', color=color)
            
            # compute top 10% weight point
            threshold = np.percentile(weights, 90)#normalized_weights, 90)
            plt.axvline(x=threshold, color=color,linestyle='--', linewidth=1, label='90th percentile' if graph_index == len(graphs)-1 else "")

    plt.title(title)
    plt.xlabel('Edge Weight')
    plt.ylabel('Frequency')
    plt.legend()
    
    # save each subplot as individual .pdf file
    plt.savefig(f"./Output/Supp_weight_distribution_{title}.pdf", format="pdf")

    plt.show()
    
def plot_degree_distribution(graphs, row_index, title):
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.get_cmap('tab10', len(graphs))  # Using a colormap for distinct colors
    
    for graph_index, graph in enumerate(graphs):
        if isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
            degrees = [degree for node, degree in graph.degree()]
            max_degree = graph.number_of_nodes()-1
            normalized_degrees = [degree / max_degree for degree in degrees]
            
            degree_counts = {degree: normalized_degrees.count(degree) for degree in set(normalized_degrees)}
            sorted_degrees = sorted(degree_counts.items())
            x, y = zip(*sorted_degrees)
            
            color = colors(graph_index)
            
            plt.plot(x, y, marker='o', color = color, label=f'Graph {graph_index+1}')
            
            threshold = 0.3
            plt.axvline(x=threshold, color='r', linestyle='--', linewidth=1, label='30th percentile' if graph_index == len(graphs)-1 else "")
 
    plt.title(title)
    plt.xlabel('Normalized Degree')
    plt.ylabel('Frequency')
    plt.legend()

    # save each subplot as individual .pdf file
    plt.savefig(f"./Output/Supp_degree_distribution_binarized_graph_{title}.pdf", format="pdf")

    plt.show()

def aggregate_node_weights(graph):
    node_weights = {}
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        
        if u not in node_weights:
            node_weights[u] = []
        node_weights[u].append(weight)

        if v not in node_weights:
            node_weights[v] = []
        node_weights[v].append(weight)
            
    
    avg_weights = {node: np.mean(weights) for node, weights in node_weights.items()}
    return avg_weights

def plot_average_weights(row_index, graphs, high_degree_hub_list):
    hub_node_avg_weights = []
    other_node_avg_weights = []

    for graph_index, graph in enumerate(graphs):
        high_hub_nodes = high_degree_hub_list[row_index][graph_index]
        
        # Aggregate average weights per node
        node_avg_weights = aggregate_node_weights(graph)

        # Separate average weights into hub nodes and other nodes
        for node in graph.nodes():
            if node in node_avg_weights:
                if node in high_hub_nodes:
                    hub_node_avg_weights.append(node_avg_weights[node])
                else:
                    other_node_avg_weights.append(node_avg_weights[node])
    # Perform Mann-Whitney U test
    stat, p_value = mannwhitneyu(hub_node_avg_weights, other_node_avg_weights, alternative='two-sided')
    
    print(f"Mann-Whitney U test results for row {row_index}: U={stat}, p-value={p_value}")
            
    plt.figure(figsize=(8, 4))

    # Plot average weights
    labels = ['High-degree Hub Nodes', 'Other Nodes']
    weights = [hub_node_avg_weights, other_node_avg_weights]

    sns.violinplot(data=weights)
    
    plt.xlabel('Node Type')
    plt.ylabel('Average Edge Weight')
    plt.title(f'Average Edge Weight Comparison for Row {row_index}')
    plt.show()
    
    return hub_node_avg_weights, other_node_avg_weights


def plot_path_lengths(row_index, graphs, high_degree_hub_list, bins=20):
    
    path_lengths_with_hub = []
    path_lengths_without_hub = []

    for graph_index, graph in enumerate(graphs):
        all_pairs_shortest_paths = dict(nx.all_pairs_bellman_ford_path_length(graph, weight='distance'))
        for source, target_lengths in all_pairs_shortest_paths.items():
            for target, length in target_lengths.items():
                if source != target:
                    shortest_path = nx.shortest_path(graph, source=source, target=target)
                    #print(shortest_path)
                    if any(node in high_degree_hub_list[row_index][graph_index] for node in shortest_path):#[1:-1]):
                        path_lengths_with_hub.append(length)
                    else:
                        path_lengths_without_hub.append(length)

    plt.figure(figsize=(8, 4))

    # Plot line graph of path lengths
    hist_with_hub, bin_edges_with_hub = np.histogram(path_lengths_with_hub, bins=bins)
    hist_without_hub, bin_edges_without_hub = np.histogram(path_lengths_without_hub, bins=bins)

    bin_centers_with_hub = (bin_edges_with_hub[:-1] + bin_edges_with_hub[1:]) / 2
    bin_centers_without_hub = (bin_edges_without_hub[:-1] + bin_edges_without_hub[1:]) / 2

    plt.plot(bin_centers_with_hub, hist_with_hub, marker='o', label='Paths through hub nodes')
    plt.plot(bin_centers_without_hub, hist_without_hub, marker='o', label='Paths without hub nodes')

    plt.xlabel('Path Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Path Length Distribution (Line Plot)')
    plt.show()
    
    # Perform Mann-Whitney U test
    stat, p_value = mannwhitneyu(path_lengths_with_hub, path_lengths_without_hub, alternative='two-sided')
    
    print(f"Mann-Whitney U test results for row {row_index}: U={stat}, p-value={p_value}")
    print(f'Shortest paths #: hub:{len(path_lengths_with_hub)} / non-hub:{len(path_lengths_without_hub)}')
    
    # Plot violin plot of path lengths
    plt.figure(figsize=(8, 4))
    data = [path_lengths_with_hub, path_lengths_without_hub]
    labels = ['Paths through hub nodes', 'Paths without hub nodes']

    sns.violinplot(data=data)
    plt.xticks(ticks=[0, 1], labels=labels)
    plt.xlabel('Path Type')
    plt.ylabel('Path Length')
    plt.title('Path Length Distribution (Violin Plot)')
    plt.show()
    
    
def plot_path_lengths_exin_merge(graphs, high_degree_hub_list, bins=20):
    all_path_lengths_with_hub = []
    all_path_lengths_without_hub = []

    for row_index, row_graphs in enumerate(graphs):
        path_lengths_with_hub = []
        path_lengths_without_hub = []

        for graph_index, graph in enumerate(row_graphs):
            all_pairs_shortest_paths = dict(nx.all_pairs_bellman_ford_path_length(graph, weight='distance'))
            for source, target_lengths in all_pairs_shortest_paths.items():
                for target, length in target_lengths.items():
                    if source != target:
                        shortest_path = nx.shortest_path(graph, source=source, target=target)
                        if any(node in high_degree_hub_list[row_index][graph_index] for node in shortest_path):
                            path_lengths_with_hub.append(length)
                        else:
                            path_lengths_without_hub.append(length)

        all_path_lengths_with_hub.append(path_lengths_with_hub)
        all_path_lengths_without_hub.append(path_lengths_without_hub)

    # Perform Mann-Whitney U test
    stat, p_value = mannwhitneyu(all_path_lengths_with_hub[1], all_path_lengths_with_hub[0], alternative='two-sided')
    print(f"Mann-Whitney U test results: U={stat}, p-value={p_value}")
    print("Excitatory_hub")
    stat, p_value = mannwhitneyu(all_path_lengths_without_hub[1], all_path_lengths_without_hub[0], alternative='two-sided')
    print(f"Mann-Whitney U test results: U={stat}, p-value={p_value}")
    print("Excitatory_non-hub")
    stat, p_value = mannwhitneyu(all_path_lengths_with_hub[3], all_path_lengths_with_hub[2], alternative='two-sided')
    print(f"Mann-Whitney U test results: U={stat}, p-value={p_value}")
    print("Inhibitory_hub")
    stat, p_value = mannwhitneyu(all_path_lengths_without_hub[3], all_path_lengths_without_hub[2], alternative='two-sided')
    print(f"Mann-Whitney U test results: U={stat}, p-value={p_value}")
    print("Inhibitory_non-hub")
 
    # Plot violin plot of path lengths
    plt.figure(figsize=(12, 4))
    
    labels = ['Paths through hub csd', 'Paths through hub local','Paths without hub csd', 'Paths without hub local']
    data = all_path_lengths_with_hub[0:2] + all_path_lengths_without_hub[0:2]
    sns.violinplot(data=data, inner="point")
    
    plt.xticks(ticks=np.arange(0,  4), labels=labels)
    plt.xlabel('Path Type')
    plt.ylabel('Path Length')
    plt.title('Excitatory Path Length Distribution (Violin Plot)')
    plt.show()

    plt.figure(figsize=(12, 4))
    
    data = all_path_lengths_with_hub[2:4] + all_path_lengths_without_hub[2:4]
    sns.violinplot(data=data, inner="point")
    
    plt.xticks(ticks=np.arange(0,  4), labels=labels)
    plt.xlabel('Path Type')
    plt.ylabel('Path Length')
    plt.title('Inhibitory Path Length Distribution (Violin Plot)')
    plt.show()
    
    plt.figure(figsize=(12, 4))
    
    
    # 위치 변경
    labels = ['Paths through hub local','Paths through hub csd','Paths without hub local','Paths without hub csd']
    data = [all_path_lengths_with_hub[1],all_path_lengths_with_hub[0]] + [all_path_lengths_without_hub[1],all_path_lengths_without_hub[0]]
    sns.violinplot(data=data, inner="point")
    
    plt.xticks(ticks=np.arange(0,  4), labels=labels)
    plt.xlabel('Path Type')
    plt.ylabel('Path Length')
    plt.title('Excitatory Path Length Distribution (Violin Plot)')
    plt.show()
    
    plt.figure(figsize=(12, 4))

    data = [all_path_lengths_with_hub[3],all_path_lengths_with_hub[2]] + [all_path_lengths_without_hub[3],all_path_lengths_without_hub[2]]
    sns.violinplot(data=data, inner="point")
    
    plt.xticks(ticks=np.arange(0,  4), labels=labels)
    plt.xlabel('Path Type')
    plt.ylabel('Path Length')
    plt.title('Inhibitory Path Length Distribution (Violin Plot)')
    plt.show()

def twogroupstat(matrix1, matrix2):
    import numpy as np
    from scipy import stats

    # 1. Levene's Test for equality of variances
    stat, p_value = stats.levene(matrix1, matrix2)
    print(f"Levene's test for equality of variances: stat={stat}, p-value={p_value}")

    # 2. Shapiro-Wilk Test for normality
    stat1, p_value1 = stats.shapiro(matrix1)
    stat2, p_value2 = stats.shapiro(matrix2)
    print(f"Shapiro-Wilk test for normality (group 1): stat={stat1}, p-value={p_value1}")
    print(f"Shapiro-Wilk test for normality (group 2): stat={stat2}, p-value={p_value2}")

    # 3. Depending on normality and variance equality, choose appropriate test
    if p_value1 > 0.05 and p_value2 > 0.05:  # If both groups are normally distributed
        if p_value > 0.05:  # If variances are equal
            t_stat, t_p_value = stats.ttest_ind(matrix1, matrix2)
            print(f"Independent t-test: t-statistic={t_stat}, p-value={t_p_value}")
            # Calculate means of both groups
            mean1 = np.mean(matrix1)
            mean2 = np.mean(matrix2)
            print(f"Mean of group 1: {mean1}")
            print(f"Mean of group 2: {mean2}")

            # Determine which group has the larger mean
            if mean1 > mean2:
                print("Group 1 has a larger mean value.")
            elif mean1 < mean2:
                print("Group 2 has a larger mean value.")
            else:
                print("Both groups have the same mean value.")
                
        else:  # If variances are not equal
            t_stat, t_p_value = stats.ttest_ind(matrix1, matrix2, equal_var=False)
            print(f"Welch's t-test: t-statistic={t_stat}, p-value={t_p_value}")
    
    else:  # If one or both groups are not normally distributed
        u_stat, u_p_value = stats.mannwhitneyu(matrix1, matrix2)
        print(f"Mann-Whitney U test: U-statistic={u_stat}, p-value={u_p_value}")
        
        # Calculate medians of both groups
        median1 = np.median(matrix1)
        median2 = np.median(matrix2)
        print(f"Mean of group 1: {median1}")
        print(f"Mean of group 2: {median2}")

        # Determine which group has the larger mean
        if median1 > median2:
            print("Group 1 has a larger median value.")
        elif median1 < median2:
            print("Group 2 has a larger median value.")
        else:
            print("Both groups have the same median value.")


        
def plot_path_lengths_in_subtype(graphs, subtype_list, high_degree_hub_list, bins=20):
    all_path_lengths_with_subtype = []
    all_path_lengths_with_hub = []
    all_path_lengths_without_hub = []

    for row_index, row_graphs in enumerate(graphs):
        path_lengths_with_subtype = []
        path_lengths_with_hub = []
        path_lengths_without_hub = []

        for graph_index, graph in enumerate(row_graphs):
            all_pairs_shortest_paths = dict(nx.all_pairs_bellman_ford_path_length(graph, weight='distance'))
            for source, target_lengths in all_pairs_shortest_paths.items():
                for target, length in target_lengths.items():
                    if source != target:
                        shortest_path = nx.shortest_path(graph, source=source, target=target)
                        if any(node in high_degree_hub_list[row_index][graph_index] for node in shortest_path):
                            path_lengths_with_hub.append(length)
                            if any(node in subtype_list[row_index][graph_index] for node in shortest_path):
                                path_lengths_with_subtype.append(length)
                        else:
                            path_lengths_without_hub.append(length)

        all_path_lengths_with_subtype.append(path_lengths_with_subtype)
        all_path_lengths_with_hub.append(path_lengths_with_hub)
        all_path_lengths_without_hub.append(path_lengths_without_hub)

        
    # Perform Mann-Whitney U test
    stat, p_value = mannwhitneyu(all_path_lengths_with_hub[1], all_path_lengths_with_hub[0], alternative='two-sided')
    print(f"Mann-Whitney U test results: U={stat}, p-value={p_value}")
    print("Inhibitory_hub")
    stat, p_value = mannwhitneyu(all_path_lengths_without_hub[1], all_path_lengths_without_hub[0], alternative='two-sided')
    print(f"Mann-Whitney U test results: U={stat}, p-value={p_value}")
    print("Inhibitory_non-hub\n\n")
        
    
    print('all_path_lengths_with_subtype:',len(all_path_lengths_with_subtype))
    print(all_path_lengths_with_subtype)
    

    print('\n**********Inhibitory csd: subtype (group 1), hub (group 2) comparison********\n')
    twogroupstat(all_path_lengths_with_subtype[0], all_path_lengths_with_hub[0])
    
    print('\n**********Inhibitory csd: subtype (group 1), nonhub (group 2) comparison**********\n')
    twogroupstat(all_path_lengths_with_subtype[0], all_path_lengths_without_hub[0])    
    
    
    print('\n**********Inhibitory local: subtype (group 1), hub (group 2) comparison**********\n')
    if len(all_path_lengths_with_subtype[1]) > 0:
        twogroupstat(all_path_lengths_with_subtype[1], all_path_lengths_with_hub[1])
    else:
        print('Skipping stat calculation as all_path_lengths_with_subtype[1] is empty.')
        
    print('\n**********Inhibitory local: subtype (group 1), nonhub (group 2) comparison**********\n')
    if len(all_path_lengths_with_subtype[1]) > 0:
        twogroupstat(all_path_lengths_with_subtype[1], all_path_lengths_without_hub[1])
    else:
        print('Skipping stat calculation as all_path_lengths_with_subtype[1] is empty.')
        
    print('\n**********Inhibitory local: subtype (group 1), csd: subtype (group 2) comparison**********\n')
    if len(all_path_lengths_with_subtype[1]) > 0:
        twogroupstat(all_path_lengths_with_subtype[1], all_path_lengths_with_subtype[0])
    else:
        print('Skipping stat calculation as all_path_lengths_with_subtype[1] is empty.')
        
    print('\n**********Inhibitory local: hub (group 1), csd: hub (group 2) comparison**********\n')
    if len(all_path_lengths_with_hub[1]) > 0:
        twogroupstat(all_path_lengths_with_hub[1], all_path_lengths_with_hub[0])
    else:
        print('Skipping stat calculation as all_path_lengths_with_hub[1] is empty.')
        
    print('////////////////////////////////////////////////////////////////////////////////////////////')

    return all_path_lengths_with_subtype
