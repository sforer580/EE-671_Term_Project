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

#include "Individual.hpp"


class EA
{
    friend class Neural_Network;
    friend class Individual;
    
public:
    Parameters* pP;
    
    void create_inputs();
    double ANN_input;
    
    vector<Individual> agent;
    void create_weights(vector<double> weights);
    void get_weights_for_ANN(int pop);
    vector<double> weights_for_ANN;
    
    void run_EA(vector<double> weights);
    
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
//creates the weigths for the ANN
void EA::create_weights(vector<double> weights)
{
    for (int pop=0; pop<pP->pop_size; pop++)
    {
        Individual I;
        agent.push_back(I);
        agent.at(pop).weights.resize(pP->num_weights);
        for (int n=0; n<pP->num_weights; n++)
        {
            agent.at(pop).weights.at(n) = pP->weight_range*((double)rand()/RAND_MAX)+pP->min_weight;
        }
    }
}


////////////////////////////////////////////////////////////////
//gets the weights for the ANN
void EA::get_weights_for_ANN(int pop)
{
    weights_for_ANN.resize(pP->num_weights);
    for (int w=0; w<pP->num_weights; w++)
    {
        weights_for_ANN.at(w) = agent.at(pop).weights.at(w);
    }
}





////////////////////////////////////////////////////////////////
//runs the entire EA
void EA::run_EA(vector<double> weights)
{
    Neural_Network ANN;
    Parameters P;
    ANN.pP = &P;
    ANN.build_ANN();
    create_inputs();
    create_weights(weights);
    for (int gen=0; gen<pP->max_gen; gen++)
    {
        for (int pop=0; pop<pP->pop_size; pop++)
        {
            get_weights_for_ANN(pop);
            ANN.run_ANN(ANN_input, weights_for_ANN);
        }
    }
}


#endif /* EA_hpp */
