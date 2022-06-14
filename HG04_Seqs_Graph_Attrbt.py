#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Microsoft VS header
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
#if os.name == 'nt' or platform == 'win32':
#    print("Running on Windows")
#    if 'ptvsd' in sys.modules:
#        print("Running in Visual Studio")
#        try:
#            os.chdir(os.path.dirname(__file__))
#            print('CurrentDir: ', os.getcwd())
#        except:
#            pass
##--------------------------------------------------#
#    else:
#        print("Running outside Visual Studio")
#        try:
#            if not 'workbookDir' in globals():
#                workbookDir = os.getcwd()
#                print('workbookDir: ' + workbookDir)
#                os.chdir(workbookDir)
#        except:
#            pass
#--------------------------------------------------#
if os.name != 'nt' and platform != 'win32':
    print("Not Running on Windows")
#--------------------------------------------------#
import sys
import time
import numpy
import pickle
import typing
import itertools
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint
from typing import Optional, Union, Tuple, Type, Set, List, Dict
#--------------------------------------------------#
import numpy as np
import pandas as pd
#--------------------------------------------------#
from PIL import Image
#--------------------------------------------------#
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# 
def generate_prot_graph(PDB_dir, save_dir, file_name = "ranked_0.pdb", threshold = 6):
    parser = PDBParser()

    for seq_idx in range(len(os.listdir(PDB_dir))):
        print("Reading sequence: ", seq_idx, "out of", len(os.listdir(PDB_dir)))
        one_AF_out_folder = os.listdir(PDB_dir)[seq_idx]
        PDB_file_location = os.path.join(PDB_dir / one_AF_out_folder, file_name)
        protein = parser.get_structure('input', PDB_file_location)
        chain = list(protein.get_chains())[0]
        #--------------------------------------------------#
        # Get residues' 3D coordinates
        residues_coordinates = []
        for res in chain:
            try:
                atom_a = res['CA'].get_coord()
                residues_coordinates.append(atom_a)
            except KeyError:
                continue
        #print("residues_coordinates: ", residues_coordinates)
        residues_coordinates = np.array(residues_coordinates)
        #--------------------------------------------------#
        # 
        tree = cKDTree(residues_coordinates)
        prot_edge = tree.sparse_distance_matrix(tree, threshold, p=2.0)
        #print("prot_edge: ", prot_edge)

        prot_edge = prot_edge.toarray()
        prot_mask = np.where(prot_edge>0, 1.0, 0.0)
        edge_dict = {}
        edge_dict['edge_feature'] = prot_edge
        edge_dict['edge_mask'] = prot_mask
        save_filename = one_AF_out_folder + "_graph_rep"
        np.save(os.path.join(save_dir, save_filename), edge_dict)
        

#######################################################################################################################################
#######################################################################################################################################
if __name__ == '__main__':
    #--------------------------------------------------#
    # I/O format
    Step_code = "HG04_"
    dataname = "phosphatase"
    # Directory and Files
    input_folder = Path("AF_out/") / (dataname + "_PDB")
    output_folder = Path("HG_results/") / (Step_code + dataname + "_seq_graph_rep")
    output_temp_folder = Path("HG_results/temp/")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_temp_folder):
        os.makedirs(output_temp_folder)   
    #--------------------------------------------------#
    generate_prot_graph(input_folder, output_folder, file_name = "ranked_0.pdb", threshold = 6)
