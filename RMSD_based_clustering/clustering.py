'''
    File name: Clustering.py
    Author: Rezvan (Leila) Chitsazi
    Python Version: 3.7.6
'''

from all_imports import *

#global data_row0_column0

def cnvt_1D_array_2D(file_name):
	
	'''VMD pairwise RMSD is 1D, needs to be converted to 2D '''

	#global data_row0_column0
	data_in=file_name
	#data_in = "rmsd-dist-matrix_10000_TMs"
	data_1D = np.loadtxt(data_in + '.dat',skiprows=0)

	dim  = data_1D.shape
	#print(dim)

	i = ''.join(str(x) for x in dim)
	dim_int = (int(i))
	print(dim_int)

	dim_sqrt = math.sqrt(dim_int)
	mat_dim = int(dim_sqrt)
	#print(dim_sqrt)

	data_2D = data_1D.reshape(mat_dim,mat_dim)
	#print(data_2D.shape)
	#print(data_2D)
	#np.savetxt('data_10000_TMs.csv', data_2D, delimiter=',')


	data_row0 = np.delete(data_2D, 0, axis= 0)
	data_row0_column0 = np.delete(data_row0, 0, axis = 1)

	#print(data_row0)
	#print(data_row0_column0)
	print('array: ', data_row0_column0.shape)

	np.savetxt('data_2d.csv', data_row0_column0, delimiter=',')

	print('')
	print('RMSD PLOT') 
	plt.imshow(data_row0_column0, cmap='viridis')
	plt.xlabel('Frame')
	plt.ylabel('Frame')
	plt.colorbar(label=r'RMSD ($\AA$)');

	return data_row0_column0		



def elbow_method(data):
	""" How to pick number of clusters (KMeans Clustering)
	 Elbow method (bending point is known as 'Elbow Point')
	within cluster sum of squared errors (WSSE)
	""" 
	k_rng = range(1,12)
	SSE = []

	for cluster in k_rng:
    		kmeans = KMeans(n_clusters = cluster, init='k-means++', algorithm='full')
    		kmeans.fit(data)
    		SSE.append(kmeans.inertia_)


	frame = pd.DataFrame({'Cluster':k_rng, 'SSE':SSE})
	plt.figure(figsize=(12,6))
	plt.plot(frame['Cluster'], frame['SSE'], marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('WSS (Inertia)');

def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)

def compute_gap(clustering, k_max=12, n_references=5):
    
    """
    Compute the Gap statistic for an nxm dataset (data)
    
    clustering: clustering method (KMeans)
    k_max:  k-values for which you want to compute the statistic (number of clusters)
    n_reference: reference distributions for automatic generation with a
    uniformed distribution within the bounding box of data
    
    """
    df = pd.read_csv('data_2d.csv', delimiter=',', header=None)
    df
    data = pd.DataFrame(df).to_numpy()
    data

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    
    """reference"""
    reference_inertia = []
    for k in range(1, k_max+1):
        local_inertia = []
        local_inertia_ = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
                                  
        reference_inertia.append(np.mean(np.log(local_inertia)))
        
    """data"""
    ondata_inertia = []
    for k in range(1, k_max+1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))
        
    gap = (reference_inertia) - np.log(ondata_inertia)
    print(gap)
    return gap, (reference_inertia), np.log(ondata_inertia)


def cluster_plot_(matrix_name, n):
    
    """
    matrix_name: matrix of the averages w/b clusters
    n: is the number of clusters selected for kmeans
    """
    
    dim = n

    mat_1D = np.loadtxt(matrix_name +'.csv')
    mat_2D = mat_1D.reshape(dim,dim)
    print(mat_2D.shape)
    print(mat_2D)

    
    clust_x = []
    clust_y = []
    
    for i in range(dim):
        
        clust_x.append("clust_" + str(i+1))
        clust_y.append("clust_" + str(i+1))    

    fig, ax = plt.subplots()
    im = ax.imshow(mat_2D, cmap='viridis')

    ax.set_xticks(np.arange(len(clust_x)))
    ax.set_yticks(np.arange(len(clust_y)))

    ax.set_xticklabels(clust_x)
    ax.set_yticklabels(clust_y)
    
    ax.grid(False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    for i in range(len(clust_x)):
        for j in range(len(clust_y)):
            text = ax.text(j, i, mat_2D[i, j],
                           ha="center", va="center", color="red")

    ax.set_title("cluster to cluster")
    fig.tight_layout()
    plt.show()


def KMeans_clustering(cluster, nframe, Lig_1, Lig_2):
    
    ''' KMeans clustering '''
    
    repeat = str(nframe) + '_TMs_'
    
    data = pd.read_csv('data_2d.csv', delimiter=',', header=None)
    X = pd.DataFrame(data).to_numpy()
    print(X.shape)
    print('')
    
    random_state = None
    init = 'random'

    kmeans = KMeans(n_clusters=cluster, init = 'random', random_state=random_state,max_iter=600, tol=0.0001)
    kmeans.fit(X)
    kmeans_pred = kmeans.predict(X)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_


    print('number of clusters is:', len(centers))
    print('number of rows belong to each cluster')

    frame = pd.DataFrame(X)
    frame['cluster'] = kmeans_pred
    frame['cluster'].value_counts()
    
    print('--------------------------------------')
    
    ''' Visualization '''
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    plt.style.use('seaborn')

    ax1.set_title("Original")
    ax1.scatter(X[:,0],X[:,3],c=X[1],cmap='brg', edgecolor=('k'))

    ax2.set_title('K_Means')
    ax2.scatter(X[:,0],X[:,3],c=kmeans.labels_,cmap='brg', edgecolor=('k'));
    
    print('-------------------------------------------------------------------------------------------')
    
    ''' Save indices for each cluster '''
    
    mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    print(mydict)

    for i in range(cluster):
        indices = list(mydict[i])
        indices_array = np.array(indices)
        np.savetxt('clust_km_'+ str(repeat) + str(i) + '.csv', indices_array.astype(int), fmt='%i')
        
    print('--------------------------------------------------------------------------------------------')   
   
    ''' distnaces within/between clusters &
        save the output in (mat_all_all_' + str(nframe) + '_TMs.csv) for
        plotting '''

    idx_ = []

    for i in range(cluster):

        idx=np.where(kmeans.labels_ == i)[0]
        idx_.append(idx)
    
    print('inter/intra')    
    print('cluster to cluster : start')
    print('')

    c1 = []
    for i in range(len(idx_)):
        c0 = []
        for j in range(len(idx_)):
            xx = np.mean(X[np.ix_(idx_[i], idx_[j])])
            print(i, j, xx)
            c0.append(xx)
        c1.append(c0)
    c1 = np.array(c1)                         
    np.savetxt('mat_all_all_' + str(nframe) + '_TMs.csv',c1.astype(float), fmt='%2.3f')
    print('')       
    print('cluster to cluster: finish')
    
    print('----------------------------------------------------')
    
   
    ''' representative clusters '''
    
    ii_ = []
    for i in range(cluster):

        ii=list(idx_[i][centers[i][idx_[i]]==min(centers[i][idx_[i]])])
        ii_.append(ii)

    print('cluster to the representative cluster: start')
    print('')

    for i in range(len(ii_)):
        for j in range(len(idx_)):
            print(np.mean(X[np.ix_(idx_[j], ii_[i])]))

    print('')
    print('cluster to the representative cluster: finish');
    
    print('-----------------------------------------------------')
    
    ''' save representative indices from each cluster
        for Pymol visualization (not fully automated, in progress)'''
    
    idx0=np.where(kmeans.labels_ == 0)[0]
    idx1=np.where(kmeans.labels_ == 1)[0]
    idx2=np.where(kmeans.labels_ == 2)[0]

    idx0_topN_Lig1=[]
    idx0_topN_Lig2=[]

    for i in range(0,100):
    
        index=list(idx0[centers[0][idx0]==(np.sort(centers[0][idx0])[i])])[0]
        if index < nframe:
            idx0_topN_Lig1.append(index)
        else:
            idx0_topN_Lig2.append(index)
            
    print('clust_1')
    L1_1 = np.unique(idx0_topN_Lig1[0:10])
    L2_1 = np.unique(idx0_topN_Lig2[0:10])
    print(Lig_1, L1_1)
    print(Lig_2, L2_1)
    np.savetxt(Lig_1 + '_rep_' + 'clust1.csv', L1_1.astype(int), fmt='%i')
    np.savetxt(Lig_2 + '_rep_' + 'clust1.csv', L2_1.astype(int), fmt='%i')
    #print(Lig_1, np.unique(idx0_topN_Lig1[0:10]))
    #print(Lig_2, np.unique(idx0_topN_Lig2[0:10]))
    #print('')
    #print(Lig_1, np.unique(idx0_topN_Lig1[0:10]))
    #print(Lig_2, np.unique(idx0_topN_Lig2[0:10]))
    print('')
    #---------------------------------------------
    idx1_topN_Lig1=[]
    idx1_topN_Lig2=[]

    for i in range(0,100):
    
        index=list(idx1[centers[1][idx1]==(np.sort(centers[1][idx1])[i])])[0]
        if index < nframe:
            idx1_topN_Lig1.append(index)
        else:
            idx1_topN_Lig2.append(index)
            
    print('clust_2')
    L1_2 = np.unique(idx1_topN_Lig1[0:10])
    L2_2 = np.unique(idx1_topN_Lig2[0:10])
    print(Lig_1, L1_2)
    print(Lig_2, L2_2)
    np.savetxt(Lig_1 + '_rep_' + 'clust2.csv', L1_2.astype(int), fmt='%i')
    np.savetxt(Lig_2 + '_rep_' + 'clust2.csv', L2_2.astype(int), fmt='%i')
    #print(Lig_1, np.unique(idx1_topN_Lig1[0:10]))
    #print(Lig_2, np.unique(idx1_topN_Lig2[0:10]))
    print('')
    #---------------------------------------------
    idx2_topN_Lig1=[]
    idx2_topN_Lig2=[]
    for i in range(0,100):
        index=list(idx2[centers[2][idx2]==(np.sort(centers[2][idx2])[i])])[0]
        if index < nframe:
            idx2_topN_Lig1.append(index)
        else:
            idx2_topN_Lig2.append(index)
            
    print('clust_3')    
    L1_3 = np.unique(idx2_topN_Lig1[0:10])
    L2_3 = np.unique(idx2_topN_Lig2[0:10])
    print(Lig_1, L1_3)
    print(Lig_2, L2_3)
    np.savetxt(Lig_1 + '_rep_' + 'clust3.csv', L1_3.astype(int), fmt='%i')
    np.savetxt(Lig_2 + '_rep_' + 'clust3.csv', L2_3.astype(int), fmt='%i')
    #print(Lig_1, np.unique(idx2_topN_Lig1[0:10]))
    #print(Lig_2, np.unique(idx2_topN_Lig2[0:10]))
    print('')

def clusters_statistics(nframe, Lig_1, Lig_2):

    files = sorted(glob.glob('clust_km_'+ str(nframe) + '*.csv'))
    print(files)        

    count = 0

    for file in range(len(files)):
        print('------------------------------------------------------------------------------------------')
        print(files[file])
        df = pd.read_csv(files[file], header=None)
        df = df.values.tolist()
        idx_ = [x[0] for x  in df]
    
    
        k = nframe
        repeat = str(nframe) + '_TMs_'
        
        k1_lst = []
        k2_lst = []
    
        count_k1 = 0
        count_k2 = 0
    
    
        for i in idx_ :
            if i > k :
            
                k1_lst.append(i)
                count_k1 = count_k1 + 1
            else :
            
                k2_lst.append(i)
                count_k2 = count_k2 +1
            
        count = count + 1 
    
        percent_k1 = (count_k1/len(idx_))*100
        percent_k2 = (count_k2/len(idx_))*100
    
        _k1_frame = (count_k1/nframe)*100
        _k2_frame = (count_k2/nframe)*100
    
        print('Total frames in cluster:', len(idx_))
        print(Lig_2 + ' frames in cluster:', count_k1, '-----',str(percent_k1) + '% of cluster total frames')
        print(Lig_1 + ' frames in cluster:', count_k2, '-----',str(percent_k2) + '% of cluster total frames')
        print('')
        print(Lig_2 + ' frames in cluster:', str(_k1_frame) + '% of total frames for RMSD, ' + str(nframe))
        print(Lig_1 + ' frames in cluster:', str(_k2_frame) + '% of total frames for RMSD, ' + str(nframe))
        print('')
    

        k1_lst=np.array(k1_lst)
        np.savetxt(Lig_2 + '_clust_' + str(repeat) +  str(count) + '.csv', k1_lst.astype(int), fmt='%i')
    
        k2_lst=np.array(k2_lst)
        np.savetxt(Lig_1 + '_clust_' + str(repeat) +  str(count) + '.csv', k2_lst.astype(int), fmt='%i')
        print('') 
