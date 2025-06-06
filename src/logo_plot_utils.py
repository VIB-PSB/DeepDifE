import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from tensorflow.keras.models import model_from_json
import tensorflow as tf #https://github.com/slundberg/shap/issues/1694
# tf.compat.v1.disable_v2_behavior() #https://github.com/slundberg/shap/issues/1694
from Bio import SeqIO
import shap

# FUNCTIONS

onecode = {'A':[1, 0, 0, 0], 'C':[0, 1, 0, 0], 'G':[0, 0, 1, 0], 'T':[0, 0, 0, 1], 'N':[0, 0, 0, 0], 'K':[0, 0, 0, 0], 'Y':[0, 0, 0, 0], 'S':[0, 0, 0, 0], 'M':[0, 0, 0, 0], 'R':[0, 0, 0, 0], 'W':[0, 0, 0, 0]}
zerocode = {'A':[0, 0, 0, 0], 'C':[0, 0, 0, 0], 'G':[0, 0, 0, 0], 'T':[0, 0, 0, 0], 'N':[0, 0, 0, 0], 'K':[0, 0, 0, 0], 'Y':[0, 0, 0, 0], 'S':[0, 0, 0, 0], 'M':[0, 0, 0, 0], 'R':[0, 0, 0, 0], 'W':[0, 0, 0, 0]}

def encode(sequence, code):
	coded_seq =  np.empty((4, len(sequence)), int) 
	for char_idx, char in enumerate(sequence): 
		coded_seq[:,char_idx] = code[char]

	return coded_seq

# Reverse complementary
def reverse_complementary(fr_samples):
	return np.flip(fr_samples, (1,2))#,2


def retextMEH(onehot):
	'''
	new more vesatile version in utils.py


	takes onehot of the form
	array([[[[0.],
		 [1.],
		 [0.],
		 ...,
		 [0.],
		 [0.],
		 [0.]],

		[[0.],
		 [0.],
		 [1.],
		 ...,
		 [0.],
		 [0.],
		 [0.]],

		[[0.],
		 [0.],
		 [0.],
		 ...,
		 [1.],
		 [0.],
		 [1.]],

		[[1.],
		 [0.],
		 [0.],
		 ...,
		 [0.],
		 [1.],
		 [0.]]],


	   [[[0.],
		 [0.],
		 [0.],
	'''
	
	seqdict = {}
	for index,letterpresence in enumerate(onehot[0]):
		if index == 0:
			for place , letter in enumerate(letterpresence):
				if letter == 1:
					seqdict[place]="A"
		elif index == 1:
			for place , letter in enumerate(letterpresence):
				if letter == 1:
					seqdict[place]="C"
		elif index == 2:
			for place , letter in enumerate(letterpresence):
				if letter == 1:
					seqdict[place]="G"
		elif index == 3:
			for place , letter in enumerate(letterpresence):
				if letter == 1:
					seqdict[place]="T"
	sorted_seqdict = {k: seqdict[k] for k in sorted(seqdict)}
	return "".join(list(sorted_seqdict.values()))

#functions from deepmel

def plot_a(ax, base, left_edge, height, color):
	a_polygon_coords = [
		np.array([
		   [0.0, 0.0],
		   [0.5, 1.0],
		   [0.5, 0.8],
		   [0.2, 0.0],
		]),
		np.array([
		   [1.0, 0.0],
		   [0.5, 1.0],
		   [0.5, 0.8],
		   [0.8, 0.0],
		]),
		np.array([
		   [0.225, 0.45],
		   [0.775, 0.45],
		   [0.85, 0.3],
		   [0.15, 0.3],
		])
	]
	for polygon_coords in a_polygon_coords:
		ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
												 + np.array([left_edge,base])[None,:]),
												facecolor=color, edgecolor=color))

def plot_c(ax, base, left_edge, height, color):
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
											facecolor=color, edgecolor=color))
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
											facecolor='white', edgecolor='white'))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
											facecolor='white', edgecolor='white', fill=True))

def plot_g(ax, base, left_edge, height, color):
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
											facecolor=color, edgecolor=color))
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
											facecolor='white', edgecolor='white'))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
											facecolor='white', edgecolor='white', fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
											facecolor=color, edgecolor=color, fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
											facecolor=color, edgecolor=color, fill=True))

def plot_t(ax, base, left_edge, height, color):
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
				  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
				  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}

def plot_weights_given_ax(ax, array,
				 height_padding_factor,
				 length_padding,
				 subticks_frequency,
				 highlight,
				 colors=default_colors,
				 plot_funcs=default_plot_funcs):
	if len(array.shape)==3:
		array = np.squeeze(array)
	assert len(array.shape)==2, array.shape
	if (array.shape[0]==4 and array.shape[1] != 4):
		array = array.transpose(1,0)
	assert array.shape[1]==4
	max_pos_height = 0.0
	min_neg_height = 0.0
	heights_at_positions = []
	depths_at_positions = []
	for i in range(array.shape[0]):
		acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
		positive_height_so_far = 0.0
		negative_height_so_far = 0.0
		for letter in acgt_vals:
			plot_func = plot_funcs[letter[0]]
			color=colors[letter[0]]
			if (letter[1] > 0):
				height_so_far = positive_height_so_far
				positive_height_so_far += letter[1]                
			else:
				height_so_far = negative_height_so_far
				negative_height_so_far += letter[1]
			plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
		max_pos_height = max(max_pos_height, positive_height_so_far)
		min_neg_height = min(min_neg_height, negative_height_so_far)
		heights_at_positions.append(positive_height_so_far)
		depths_at_positions.append(negative_height_so_far)
	for motif, info in highlight.items(): 
		for (start_pos, end_pos) in info["startstops"]:          
			assert start_pos >= 0.0 and end_pos <= array.shape[0]
			min_depth = np.min(depths_at_positions[start_pos:end_pos])
			max_height = np.max(heights_at_positions[start_pos:end_pos])
			ax.text(start_pos,max_height,motif)
			ax.add_patch(
				matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
					width=end_pos-start_pos,
					height=max_height-min_depth,
					edgecolor=info["color"],
					fill=False,
					label="test"))
		
	ax.set_xlim(-length_padding, array.shape[0]+length_padding)
	ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
	height_padding = max(abs(min_neg_height)*(height_padding_factor),
						 abs(max_pos_height)*(height_padding_factor))
	ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
	return ax

def plot_weights_modified(array, fig, n,n1,n2, title='', ylab='',
							  figsize=(20,2),
				 height_padding_factor=0.2,
				 length_padding=1.0,
				 subticks_frequency=20,
				 colors=default_colors,
				 plot_funcs=default_plot_funcs,
				 highlight={}):
	ax = fig.add_subplot(n,n1,n2) 
	ax.set_title(title)
	ax.set_ylabel(ylab)
	y = plot_weights_given_ax(ax=ax, array=array,
		height_padding_factor=height_padding_factor,
		length_padding=length_padding,
		subticks_frequency=subticks_frequency,
		colors=colors,
		plot_funcs=plot_funcs,
		highlight=highlight)
	return fig,ax
