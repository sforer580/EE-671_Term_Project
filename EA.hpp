//
//  EA.hpp
//  EE-671_Term_Project
//
//  Created by Scott S Forer on 10/16/16.
//  Copyright © 2016 Scott S Forer. All rights reserved.
//

#ifndef EA_hpp
#define EA_hpp

#include <stdio.h>



class EA
{
    friend class Neural_Network;
    
public:
    Parameters* pP;
    
    void create_inputs();
    double ANN_input;
    
    void run_EA();
    
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ANN inputs

////////////////////////////////////////////////////////////////
//creates the inputs for the ANN
void EA::create_inputs()
{
    for (int input=0; input<pP->input_layer_size; input++)
    {
        ANN_input = 0;
        double max = pP->input_upper_limit;
        double min = pP->input_lower_limit;
        double range = max-min;
        //cout << range << endl;
        ANN_input = range*((double)rand()/RAND_MAX)+min;
    }
}





////////////////////////////////////////////////////////////////
//creates the weights for the ANN





void EA::run_EA()
{
    Neural_Network ANN;
    Parameters P;
    ANN.pP = &P;
    ANN.build_ANN();
    create_inputs();
    ANN.run_ANN(ANN_input);
}


#endif /* EA_hpp */
