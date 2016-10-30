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
    
    void build_pop();
    void create_inputs();
    double ANN_input;
    
    vector<Individual> agent;
    void create_weights(vector<double> weights);
    void get_weights_for_ANN(int pop);
    vector<double> weights_for_ANN;
    
    void get_target();
    double target;
    void get_fitness(int pop);
    void natural_selection();
    int binary_select();
    void mutation(Individual &M);
    
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
void EA::build_pop()
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
//gets the target fucntion output
void EA::get_target()
{
    target = 0;
    target = ANN_input*ANN_input + 1;
}


////////////////////////////////////////////////////////////////
//gets the fitness for an agent
void EA::get_fitness(int pop)
{
    agent.at(pop).fitness = abs(target - agent.at(pop).agent_output);
}


////////////////////////////////////////////////////////////////
//randomly selects two individuals and decides which one will die based on their fitness
int EA::binary_select()
{
    int loser;
    int index_1 = rand() % agent.size();
    int index_2 = rand() % agent.size();
    while (index_1 == index_2)
    {
        index_2 = rand() % agent.size();
    }
    if(agent.at(index_1).fitness > agent.at(index_2).fitness)
    {
        loser = index_2;
    }
    else
    {
        loser = index_1;
    }
    return loser;
}


////////////////////////////////////////////////////////////////
//mutates the copies of the winning individuals
void mutation(Individual &M)
{
    
    
 
    
}


////////////////////////////////////////////////////////////////
//runs the entire natural selectiona dn mutation process
void EA::natural_selection()
{
    int kill;
    for(int k=0; k<pP->to_kill; k++)
    {
        kill = binary_select();
        agent.erase(agent.begin() + kill);
    }
    
    int to_replicate = pP->to_kill;
    for (int rRrR=0; rRrR<to_replicate; rRrR++)
    {
        Individual M;
        int spot = rand() % agent.size();
        M = agent.at(spot);
        cout << "cp" << endl;
        mutation(M);
        //agent.push_back(M);
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
    build_pop();
    for (int gen=0; gen<pP->max_gen; gen++)
    {
        create_inputs();
        get_target();
        for (int pop=0; pop<pP->pop_size; pop++)
        {
            get_weights_for_ANN(pop);
            agent.at(pop).agent_output = ANN.run_ANN(ANN_input, weights_for_ANN);
            get_fitness(pop);
        }
        natural_selection();
        
    }
}


#endif /* EA_hpp */
