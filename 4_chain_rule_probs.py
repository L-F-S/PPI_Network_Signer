# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:30:48 2023

@author:L-F-S

# to calculate output sign as probability of being -,
we have to take into account all (unique) permutations
of odd number of - in the pathway
i.e. for a 5 step pathway,
there are permutations for 1, 3, and 5 minuses per pathway
we sum the probabilities of all them to get the probability of a - output sign

i.e. for a 5-step pathway,
with given probability of negative edge
p_minus = [0.99,0.01,0.01,0.4,0.4]
one needs to find all combinations of signs with odd number of negative
edges, here are all those with 3 negative edges:
    
'''
{('+', '+', '-', '-', '-'),
 ('+', '-', '+', '-', '-'),
 ('+', '-', '-', '+', '-'),
 ('+', '-', '-', '-', '+'),
 ('-', '+', '+', '-', '-'),
 ('-', '+', '-', '+', '-'),
 ('-', '+', '-', '-', '+'),
 ('-', '-', '+', '+', '-'),
 ('-', '-', '+', '-', '+'),
 ('-', '-', '-', '+', '+')}

#associate the given probability at each position, and multiply all rows (items in tuple),
# and sum all 

Do the same with all permutation with 1 negative edge, and 5 (which is all negative)
and sum all
'''
"""

import itertools
import numpy as np
import random



#example
# show positions to have - probability in the set, to have odd number of 
# - signs
# the list should have 2 corresponding lists; one of positive and one fo 
# negative probs:


def chain_rule(p_minus):
# find all odd indexes <= len(p_minus)

    p_minus=np.array(p_minus)
    p_plus=np.array([1-p for p in p_minus])
    cum_prob_old_method=0
    cum_prob_new_method=0
    n_steps = len(p_minus)
    odd_nums=[number for number in range(n_steps + 1) if number % 2 != 0]
    # odd_nums=[1,3,5]
    for odd_num in odd_nums: 
        # find all combinations of steps with ind (i.e. 3 minuses
        
        minus_list = np.array([-1]*odd_num+[+1]*(n_steps-odd_num))
        #minus_list = [-1, -1, -1, 1, 1]
        
        #################################
        #find cobinations (OLD METHOD)
        #################################
        current_combs = set(itertools.permutations(minus_list,n_steps))
        
        curr_cum_prob_old=0
        for signs in current_combs:
            probs = np.where(np.array(signs)==-1, p_minus, p_plus)
            probs=np.prod(probs)
            curr_cum_prob_old+=probs
        
        
        ##############################################################
        # find combinations of odd indexes for minus edge: [NEW METHOD]
        ##############################################################
        minus_inds = itertools.combinations(range(len(p_minus)),odd_num)
        
        curr_cum_prob_new = 0
        for indexes in minus_inds:
            probs=p_plus.copy()
            probs[list(indexes)] = p_minus[list(indexes)]
            probs=np.prod(probs)
            curr_cum_prob_new+=probs
            
            
        cum_prob_old_method += curr_cum_prob_old
        cum_prob_new_method += curr_cum_prob_new
    return cum_prob_old_method, cum_prob_new_method
  
# run function for 10 5-arrays of random numbers between 0 and 1:
for random_path in [[random.uniform(0,1) for i in range(5)] for random_prob_array in range(10)]:
    print(chain_rule(random_path))