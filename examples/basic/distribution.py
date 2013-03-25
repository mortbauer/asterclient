# coding=utf-8
from math import *

def make_group_liaisons(group1,group2,dofs):
    lias = []
    for dof in dofs:
        lias.append({'GROUP_NO_1':group1 ,'GROUP_NO_2':group2,'DDL_1':dof,'DDL_2':dof,'COEF_MULT_1':1,'COEF_MULT_2':-1,'COEF_IMPO':0})
    return lias


parameters = (('lenkkopfsteifigkeit',{
    'beam_parts':['forklong'],
    'bar_parts':[],
    'boundaries': [
        {'GROUP_NO':('p_f_0_l','p_f_0_r'),'DX':0,'DY':0,'DZ':0,'DRX':0,'DRY':0,'DRZ':0},
        {'GROUP_NO':('p_f_3a'),'DX':0}],
    'liaisons': make_group_liaisons('p_f_3','p_fl_3',['DX','DY','DZ','DRZ']) +
                    make_group_liaisons('p_f_4','p_fl_4',['DX','DY','DZ','DRZ']),
    'liaisons_solide': [],
    'forces':[{'GROUP_NO':'p_fl_5','FX':-100}],
    'displacement_nodes':['p_fl_7'],
    'study_type':'linear'}),

        ('tretlagersteifigkeit',{
    'beam_parts':['forkshor','adap45','backsu','forkadap'],
    'bar_parts': ['kettena'],
    'boundaries':[
            {'GROUP_NO':('p_fsa_6a'),'DX':0,'DY':0,'DZ':0,'DRX':0,'DRY':0,'DRZ':0},
            {'GROUP_NO':('p_b_0a'),'DX':0,'DY':0,'DZ':0}],
    'liaisons': make_group_liaisons('p_fs_6','p_fsa_6',['DX','DY','DZ','DRZ','DRY','DRX']),
    'liaisons_solide': [{'GROUP_NO':x} for x in [('p_f_0_l','p_b_0_l'),('p_f_3','p_fs_3'),
            ('p_f_4','p_fs_4'),('p_f_1_l','p_a_1_l'),('p_f_0_r','p_b_0_r','p_k_0_r'),('p_a_1b_r','p_k_1b_r')]],
    'forces':[ {'GROUP_NO':'p_a_1e_r','FZ':-1000 * sin(radians(82.5)), 'FX': -1000 * cos(radians(82.5))}],
    'displacement_nodes': ['p_a_1e_r'] ,
    'study_type':['linear']})
        )
