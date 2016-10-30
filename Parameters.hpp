//
//  Parameters.hpp
//  EE-671_Term_Project
//
//  Created by Scott S Forer on 10/16/16.
//  Copyright © 2016 Scott S Forer. All rights reserved.
//

#ifndef Parameters_hpp
#define Parameters_hpp

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <iomanip>



class Parameters
{
    friend class Neural_Network;
    friend class Layer;
    friend class Node;
    friend class EA;
    
public:
    int num_layers = 3;         //must be 3
    int input_layer_size = 1;   //must be 1
    int hidden_layer_size = 2;
    int output_layer_size = 1;  //myst be 1
    
    double input_upper_limit = 5;
    double input_lower_limit = -5;
    double output_upper_limit = 30;
    double output_lower_limit = -2;
    
    int num_i_h_w = (input_layer_size+1)*(hidden_layer_size);
    int num_h_o_w = (hidden_layer_size+1)*(output_layer_size);
    int num_weights = num_i_h_w + num_h_o_w;
    double max_weight = 1;
    double min_weight = -1;
    double weight_range = max_weight-min_weight;
    
    
    int pop_size = 10;
    int to_kill = pop_size/2;
    double mutation_rate = 0.5;
    double range = 0.01;
    int max_gen = 100;
    
    
    
};




#endif /* Parameters_hpp */
