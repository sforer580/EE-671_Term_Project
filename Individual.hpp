//
//  Individual.hpp
//  EE-671_Term_Project
//
//  Created by Scott S Forer on 10/16/16.
//  Copyright © 2016 Scott S Forer. All rights reserved.
//

#ifndef Individual_hpp
#define Individual_hpp

#include <stdio.h>


class Individual
{
    friend class Neural_Network;
    friend class EA;
    
public:
    vector<double> weights;
    double fitness;
};


#endif /* Individual_hpp */
