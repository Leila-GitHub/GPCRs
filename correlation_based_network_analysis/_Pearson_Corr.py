'''
    File name: _Pearson_Corr.py
    Author: Rezvan (Leila) Chitsazi
    Python Version: 3.7.6
'''

import os, sys, glob
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import re

import matplotlib.image as mpimg

pd.set_option("display.max_rows", None, "display.max_columns", None)
#*******************************

def set_colour(x):
    if x > 0.75:
        color = 'green'
    elif x < -0.75:
        color = 'red'
    else:
        color = 'black'
    return 'color: %s' % color


def _r_(sample, nrows, ncols, corr_threshold, which=True):

 
    """Pearson's Correlation : r"""
    """_sample_.csv: input file (e.g. _mdmb_.csv)"""
    sample_ = pd.read_csv("_"+sample+"_.csv", nrows=nrows, usecols=range(ncols))

    """Pearson Correlation"""
    sample_corr_result = sample_.corr(method ='pearson')  
    sample_corr_result.to_csv(sample+'_corr_matrix.csv')

    sample_corr_style = sample_corr_result.style.applymap(set_colour).format("{:.2f}")
   
    """Filtering""" 
    corr_result = sample_corr_result.stack()
    sample_result = corr_result[(corr_result != 1.0)&((corr_result > corr_threshold)|(corr_result < -corr_threshold))]

    sample_result.to_csv(sample+'_result.csv') 

    _sample_ = pd.read_csv(sample+"_result.csv") 
    _sample_.columns=['index_1','index_2', 'corr']

    _sample_.to_csv(sample+'_filtered.csv', index=False)

    """Un-duplicating""" 
    file_name = sample+"_filtered.csv"
    file_name_output = sample+"_unduplicated.csv"

    df = pd.read_csv(file_name, sep=",")
    df.drop_duplicates(subset=['corr'], inplace=True)
    df.to_csv(file_name_output,index=False)

    if which is True:
        return (sample_corr_style)
    else:
        return (sample_.head(5))

    return

def _common_diff(sample1, sample2, name, which=None):

    df1 = pd.read_csv(sample1+'_unduplicated.csv', usecols = ['index_1','index_2'], low_memory = True)
    df2 = pd.read_csv(sample2+'_unduplicated.csv', usecols = ['index_1','index_2'], low_memory = True)

    """Find rows which are different/common between two DataFrames."""

    """
    which='both' 
    sample1: which='left_only'
    sample2: which='right_only'
    """      

    comparison_df = df1.merge(
        df2,
        indicator=True,
        how='outer'
    )
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both'];
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]

    diff_df.to_csv(name+'_diff_common.csv')
    return diff_df 

def _group_(name, keyword1, keyword2):

     df = pd.read_csv(name+'_diff_common.csv', usecols = ['index_1','index_2'], low_memory = True)
     out = df[df['index_1'].str.contains(keyword1) & df['index_2'].str.contains(keyword2)]
     
     return (out)

def _Corr_Plotting(sample1, sample2, corr_threshold):

    sample1_ = pd.read_csv(sample1+"_corr_matrix.csv", index_col=0)
    sample2_ = pd.read_csv(sample2+"_corr_matrix.csv", index_col=0)
  
    sns.set(font_scale=1.4)

    sample1_[np.abs(sample1_)<corr_threshold] = 0
    sample2_[np.abs(sample2_)<corr_threshold]  = 0

    fig, ax =plt.subplots(1,2, figsize=(75,30))
    #fig, ax = plt.subplots(1,2)

    sns.heatmap(sample1_, ax=ax[0], vmin=-1, cmap='coolwarm',annot=False, linewidths=0.5, xticklabels=False, yticklabels=False)
    sns.heatmap(sample2_, ax=ax[1], vmin=-1, cmap='coolwarm',annot=False, linewidths=0.5, xticklabels=False, yticklabels=False)
 
    plt.show()

def _Corr_diff_Plotting(sample1, sample2, corr_threshold):
   
    sample1_ = pd.read_csv(sample1+"_corr_matrix.csv", index_col=0)
    sample2_ = pd.read_csv(sample2+"_corr_matrix.csv", index_col=0) 

    diff_matrix = sample1_.subtract(sample2_) 
    diff_matrix[np.abs(diff_matrix)<corr_threshold] = 0

    mask = np.triu(np.ones_like(diff_matrix, dtype=bool))

    fig, ax =plt.subplots(figsize=(18,16))

    heatmap = sns.heatmap(diff_matrix, mask = mask, ax=ax, vmin=-1, vmax=1, cmap='vlag', annot=False, linewidths=0.5)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)

    plt.show()    

def plotimages(Lig_name):
    images = []
    for img_path in sorted(glob.glob('*' + Lig_name + '.tiff')):
        images.append(mpimg.imread(img_path))

    print('The number of pairs for ', Lig_name, 'is:' ,len(images))
    plt.figure(figsize=(40,len(images)+10))

    
    columns = 6
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.axis('off')
        plt.imshow(image) 




#def plotimages(n):
 #   images = []
 #   for img_path in sorted(glob.glob('*.tiff')):
  #      images.append(mpimg.imread(img_path))
#
#    plt.figure(figsize=(90,65))
#
#    columns = n
#    for i, image in enumerate(images):
 #       plt.subplot(len(images) / columns + 1, columns, i + 1)
 #       plt.axis('off')
  #      plt.imshow(image)  

def my_function(a):
    for x in range (len(a)):
        if a[x] < 0:
            a[x] = a[x] + 360
    else:
            a[x] = a[x]


def unique(list1):

    unique_list = []

    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def filtering_(Lig_name,whichone):

	print(Lig_name)
	usecols=[0,1,2,3,4,5,14,15]
	filtered = pd.read_csv(Lig_name + '_residue_filtered.csv', usecols = usecols,delimiter=",") 
	print(filtered.shape[0])
	print(filtered.head(5))

	filtered_sorted = filtered.sort_values(['index_1','diff_residue1','diff_residue2'], ascending =[True,False,False])
	print(filtered_sorted.shape[0])
	filtered_sorted.head(5)

	duplicated_droped_index1 = filtered_sorted.drop_duplicates(subset=['index_1','TM_residue1','TM_residue2'], keep='first')
	print(duplicated_droped_index1.shape[0])
	duplicated_droped_index1.head(5)

	duplicated_droped_index2 = duplicated_droped_index1.drop_duplicates(subset=['index_2','TM_residue1','TM_residue2'], keep='first')
	duplicated_droped_index2.to_csv(Lig_name + '_duplicated_dropped_index.csv', index=False)
	print(duplicated_droped_index2.shape[0])


	sorted_ = duplicated_droped_index2.sort_values(by=['TM_residue1','TM_residue2','corr'], ascending=[False,False,False])
	sorted_.index = np.arange(0, len(sorted_))
	sorted_.to_csv(Lig_name + '_' + whichone + '_final_pairs.csv', index=False)
	sorted_
     
	'''result_ = sorted_.groupby(['TM_residue1','TM_residue2'], as_index=True).agg({'corr': ['min', 'max']})'''

	subgroup_result_ = sorted_.groupby(['TM_residue1','TM_residue2'], as_index=True).agg({'corr': ['count']})
	print (subgroup_result_)


	layer_name_1 = duplicated_droped_index2.index_1.str.extract('.*\((.*)\).*')
	layer_name_2 = duplicated_droped_index2.index_2.str.extract('.*\((.*)\).*')

	list1 = (layer_name_1.values.tolist())
	list2 = (layer_name_2.values.tolist())


	a = re.sub('[\[\]]','',repr(unique(list1)))
	b = re.sub('[\[\]]','',repr(unique(list2))) 

	print(a)
	print(b)
	print('')

#*****************************************************************************************

	index_1 = [['L1','L1','None'],['L1','L2','L4'],['L1','L3','None'],['L1','L4','None'],
           ['L2','L2','L3'],['L2','L2-L3','None'],['L2-L3','L2','None'],
           ['L2','L4','L3'],['L3','L3-L4','L2'],
           ['L2-L3','L3','L4'],['L3','L2-L3','L4'],['L4','L3','L2'],['L4','L4','L2']]

	for lst in index_1:
        
   	 filtered_ = (duplicated_droped_index2[(duplicated_droped_index2['index_1'].str.contains(lst[0])==True)  & 
                                   (duplicated_droped_index2['index_2'].str.contains(lst[1])==True)  & 
                                   (duplicated_droped_index2['index_1'].str.contains(lst[2])==False) & 
                                   (duplicated_droped_index2['index_2'].str.contains(lst[2])==False)]) 
   	 if not filtered_.empty:
            
        	filtered_header = filtered_[['index_1','index_2','corr','TM_residue1','TM_residue2']]
        
        	print('')
        	print(filtered_header)

	return filtered_header


def _2d_map_corr_(data,lst,label_x,label_y, col_1, col_2,_save):
        
    _Lig = pd.read_csv('__'+ data + '__.csv', header=None, delimiter=",", usecols=range(sample_ncols))
         
    plt.figure(figsize=(2,2))
    sns.set(rc={'axes.labelsize':  15,
                'xtick.labelsize': 16,
                'figure.figsize':  (6, 4),
                'ytick.labelsize': 16,
                'figure.dpi' : 300, 
                'font.weight': 'bold'})
   
    
    d1 = _Lig[col_1]
    my_function(d1)
    x = d1
    y = _Lig[col_2]

    df = pd.DataFrame(_Lig.values, columns=lst)
    
    
    
    xlim_1=df.min()[label_x]
    xlim_1=xlim_1-0.5
    
    xlim_2=df.max()[label_x]
    xlim_2=xlim_2+0.5
    
    ylim_1=df.min()[label_y]
    ylim_1=ylim_1-0.5
    
    ylim_2=df.max()[label_y]
    ylim_2=ylim_2+0.5
    
    print(label_x, label_y)

    if data=='mdmb':
        
        h=sns.jointplot(x=label_x, y=label_y, data=df, kind="kde",
                      stat_func=None, space=0.2, xlim=[xlim_1, xlim_2], ylim=[ylim_1,ylim_2],
                      shade=True, shade_lowest=False, marginal_kws={'lw': 2, 'color': 'brown'});
        h.ax_joint.set_xlabel(label_x, fontweight='bold')
        h.ax_joint.set_ylabel(label_y, fontweight='bold')
        plt.savefig(_save + '_mdmb.tiff')
        plt.close()
        
    else:
        
        hh=sns.jointplot(x=label_x, y=label_y, data=df, kind="kde",
                      stat_func=None, space=0.2, xlim=[xlim_1, xlim_2], ylim=[ylim_1,ylim_2],
                      shade=True, shade_lowest=False, marginal_kws={'lw': 2, 'color': 'cyan'});
        hh.ax_joint.set_xlabel(label_x, fontweight='bold')
        hh.ax_joint.set_ylabel(label_y, fontweight='bold')
        plt.savefig(_save + '_mmb.tiff')
        plt.close()


def read_final_list_and_plot_2d_map(Lig_name, whichone):
    
    """both: common between mdmb & mmb
       only: pairs only for mdmb or mmb
       both: number for pairs should be the same for mdmb & mmb
       both: corr. coefficients are diff between mdmb and mmb
    """
    
    global sample_ncols
    sample = pd.read_csv('_' + Lig_name + '_.csv')
    sample_nrows = sample.shape[0]
    sample_ncols = sample.shape[1]

    header = sample.columns.tolist()
    print('Number of initial distances (input) for ' + Lig_name + ' is: ' + str(sample.shape[1]))

    final_pairs = pd.read_csv(Lig_name +'_' + whichone + '_final_pairs.csv')
    final_pairs
   
    pairsList = []
    for index, row in final_pairs.iterrows(): 
        mylist = [row.index_1, row.index_2]
        pairsList.append(mylist)
    pairsList

    count = 0
    for lst in pairsList:
    
        col_1 = 0
        col_2 = 1

        label_x =lst[0]
        label_y =lst[1]

        count = count + 1
        _save1 ='final_' + str(count) + '_' + whichone

        _2d_map_corr_(Lig_name,header, label_x, label_y, col_1, col_2,_save1)
