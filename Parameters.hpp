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
    friend class EA;
    
public:
    int num_layers = 3;         //must be 3
    int input_layer_size = 1;   //must be 1
    int hidden_layer_size = 10;
    int output_layer_size = 1;  //must be 1
    
    double input_upper_limit;
    double input_lower_limit;
    double output_upper_limit;
    double output_lower_limit;
    
    //ANN parameters
    int num_i_h_w = (input_layer_size+1)*(hidden_layer_size);
    int num_h_o_w = (hidden_layer_size+1)*(output_layer_size);
    int num_weights = num_i_h_w + num_h_o_w;
    double max_weight = 1;
    double min_weight = -1;
    double weight_range = max_weight-min_weight;
    
    //EA Parameters
    int pop_size = 100;
    int to_kill = pop_size/2;
    double mutation_rate = 0.5;
    double range = 0.5;
    int max_gen = 500;
    int num_tp = 100;
    
    //experimental parameters
    int x_squared_plus_1_minus_5_0 = 0;     //0=off, 1=on
    int x_squared_plus_1_0_5 = 1;           //0=off, 1=on
    int sine_0_pi_div_2 = 0;                //0=off, 1=on
    int sine_pi_div_2_pi = 0;               //0=off, 1=on
    int sine_pi_3_pi_div_2 = 0;             //0=off, 1=on
    int sine_3_pi_div_2_2_pi = 0;           //0=off, 1=on
    int cos_0_pi_div_2 = 0;                 //0=off, 1=on
    int cos_pi_div_2_pi = 0;                //0=off, 1=on
    int cos_pi_3_pi_div_2 = 0;              //0=off, 1=on
    int cos_3_pi_div_2_2_pi = 0;            //0=off, 1=on
    int tan_minus_pi_div_2_0 = 0;           //0=off, 1=on
    int tan_0_pi_div_2 = 0;                 //0=off, 1=on
};




#endif /* Parameters_hpp */
