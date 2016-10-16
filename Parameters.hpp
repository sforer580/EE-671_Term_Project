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
    int input_layer_size = 1;
    int hidden_layer_size = 1;
    int output_layer_size = 1;
    
    double input_upper_limit = 2;
    double input_lower_limit = -2;
    double output_upper_limit = 2;
    double output_lower_limit = -2;
    
    
    
};




#endif /* Parameters_hpp */
