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
#include <fstream>

#include "Individual.hpp"


class EA
{
    friend class Neural_Network;
    friend class Individual;
    
public:
    Parameters* pP;
    
    void build_pop();
    void create_inputs();
    vector <double> inputs;
    double ANN_input;
    
    vector<Individual> agent;
    void create_weights(vector<double> weights);
    void get_weights_for_ANN(int pop);
    vector<double> weights_for_ANN;
    
    void get_target();
    double target;
    void get_error(int pop);
    void get_fitness(int pop);
    void natural_selection();
    int binary_select();
    void mutation(Individual &M);
    
    void sort_indivduals_fitness();
    struct less_than_agent_fitness;
    void get_statistics();
    vector<double> min_fitness;
    void write_statistics_to_file();
    
    void get_best_agent();
    void write_best_weigths_to_file(Individual &BI);
    void get_best_weights_for_ANN(Individual &BI);
    vector<double> best_input;
    vector<double> best_output;
    vector<double> target_output;
    void write_best_inputs_and_outputs_to_file();
    void write_target_inputs_and_outputs_to_file();
    void run_best_agent(Individual &BI);
    
    void run_EA(vector<double> weights);
    
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ANN inputs

////////////////////////////////////////////////////////////////
//creates the inputs for the ANN
void EA::create_inputs()
{
    inputs.clear();
    inputs.resize(pP->num_tp);
    for (int tp=0; tp<pP->num_tp; tp++)
    {
        for (int input=0; input<pP->input_layer_size; input++)
        {
            ANN_input = 0;
            double max = pP->input_upper_limit;
            double min = pP->input_lower_limit;
            double range = max-min;
            //cout << range << endl;
            inputs.at(tp) = range*((double)rand()/RAND_MAX)+min;
            //ANN_input = range*((double)rand()/RAND_MAX)+min;
        }
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
        
        //cout << "agent" << "\t" <<  pop << "\t" << "weights" << endl;
        for (int n=0; n<pP->num_weights; n++)
        {
            agent.at(pop).weights.at(n) = pP->weight_range*((double)rand()/RAND_MAX)+pP->min_weight;
            //cout << agent.at(pop).weights.at(n) << "\t";
        }
        //cout << endl;
        
    }
}


////////////////////////////////////////////////////////////////
//gets the weights for the ANN
void EA::get_weights_for_ANN(int pop)
{
    weights_for_ANN.clear();
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
    //target = ANN_input*ANN_input + 1;
    target = cos(ANN_input);
    //target = sin(ANN_input);
}


////////////////////////////////////////////////////////////////
//gets the error for an agent
void EA::get_error(int pop)
{
    agent.at(pop).error += abs(target - agent.at(pop).agent_output);
}


////////////////////////////////////////////////////////////////
//gets the fitness for an agent
void EA::get_fitness(int pop)
{
    agent.at(pop).fitness = agent.at(pop).error;
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
    if(agent.at(index_1).fitness < agent.at(index_2).fitness)
    {
        loser = index_2;
        //cout << "loser" << "\t" <<  "agent" << "\t" << index_2 << endl;
    }
    else
    {
        loser = index_1;
        //cout << "loser" << "\t" <<  "agent" << "\t" << index_1 << endl;
    }
    return loser;
}


////////////////////////////////////////////////////////////////
//mutates the copies of the winning individuals
void EA::mutation(Individual &M)
{
    for (int w=0; w<pP->num_weights; w++)
    {
        double random = ((double)rand()/RAND_MAX);
        if (random <= pP->mutation_rate)
        {
            double R1 = ((double)rand()/RAND_MAX) * pP->range;
            double R2 = ((double)rand()/RAND_MAX) * pP->range;
            M.weights.at(w) = M.weights.at(w) + (R1-R2);
        }
    }
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
        //cout << "cp" << endl;
        mutation(M);
        agent.push_back(M);
        agent.at(agent.size()-1).age = 0;
    }
}


/////////////////////////////////////////////////////////////////
//sorts the population based on their fitness from lowest to highest
struct EA::less_than_agent_fitness
{
    inline bool operator() (const Individual& struct1, const Individual& struct2)
    {
        return (struct1.fitness < struct2.fitness);
    }
};



////////////////////////////////////////////////////////////////
//sorts population
void EA::sort_indivduals_fitness()
{
    for (int indv=0; indv<pP->pop_size; indv++)
    {
        sort(agent.begin(), agent.end(), less_than_agent_fitness());
    }
}


////////////////////////////////////////////////////////////////
//gets the statistics for each generation
void EA::get_statistics()
{
    min_fitness.push_back(agent.at(0).fitness);
}


////////////////////////////////////////////////////////////////
//writes the statistics to a txt file
void EA::write_statistics_to_file()
{
    ofstream File1;
    File1.open("Min_fitness.txt");
    for (int i=0; i<min_fitness.size(); i++)
    {
        File1 << min_fitness.at(i) << endl;
    }
}


////////////////////////////////////////////////////////////////
//Gets the best agent after the EA is complete
void EA::write_best_weigths_to_file(Individual &BI)
{
    ofstream File2;
    File2.open("Best_weights.txt");
    for (int w=0; w<pP->num_weights; w++)
    {
        File2 << agent.at(0).weights.at(w) << endl;
    }
}


////////////////////////////////////////////////////////////////
//gets best the agents weights for the ANN
void EA::get_best_weights_for_ANN(Individual &BI)
{
    weights_for_ANN.clear();
    weights_for_ANN.resize(pP->num_weights);
    for (int w=0; w<pP->num_weights; w++)
    {
        weights_for_ANN.at(w) = BI.weights.at(w);
        //cout << weights_for_ANN.at(w) << "\t";
    }
    //cout << endl;
}


////////////////////////////////////////////////////////////////
//Runs best agent after the EA is complete
void EA::run_best_agent(Individual &BI)
{
    Neural_Network ANN;
    Parameters P;
    ANN.pP = &P;
    ANN.build_ANN();
    double tp = pP->input_lower_limit;
    while (tp<pP->input_upper_limit)
    {
        best_input.push_back(tp);
        ANN_input = tp;
        target_output.push_back(ANN_input*ANN_input+1);
        //cout << best_input.at(tp) << endl;
        //cout << "input" << "\t" << ANN_input << endl;
        get_best_weights_for_ANN(BI);
        BI.agent_output = ANN.run_ANN(ANN_input, weights_for_ANN);
        best_output.push_back(BI.agent_output);
        //cout << "agent output" << "\t" << agent.at(pop).agent_output << endl;
        tp += 0.1;
    }
}


////////////////////////////////////////////////////////////////
//Writes the best inputs and outputs to a txt file
void EA::write_best_inputs_and_outputs_to_file()
{
    ofstream File3;
    File3.open("Best_input_and_output");
    for (int tp=0; tp<best_input.size(); tp++)
    {
        File3 << best_input.at(tp) << "\t" << best_output.at(tp) << endl;
    }
}


////////////////////////////////////////////////////////////////
//Writes the target inputs and outputs to a txt file
void EA::write_target_inputs_and_outputs_to_file()
{
    ofstream File4;
    File4.open("Target_input_and_output");
    for (int tp=0; tp<best_input.size(); tp++)
    {
        File4 << best_input.at(tp) << "\t" << target_output.at(tp) << endl;
    }
}


////////////////////////////////////////////////////////////////
//Gets the best agent after the EA is complete
void EA::get_best_agent()
{
    Individual BI;
    BI = agent.at(0);
    write_best_weigths_to_file(BI);
    run_best_agent(BI);
    write_best_inputs_and_outputs_to_file();
    write_target_inputs_and_outputs_to_file();
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
        cout << "------------------------------------------------------------" << endl;
        cout << "generation" << "\t" << gen << endl;
        for (int pop=0; pop<pP->pop_size; pop++)
        {
            agent.at(pop).agent_output = 0;
            agent.at(pop).fitness = 0;
            agent.at(pop).error = 0;
            //cout << "agent" << "\t" << pop << endl;
            for (int tp=0; tp<pP->num_tp; tp++)
            {
                ANN_input = inputs.at(tp);
                //cout << "input" << "\t" << ANN_input << endl;
                get_target();
                //cout << "target" << "\t" << target << endl;
                get_weights_for_ANN(pop);
                agent.at(pop).agent_output = ANN.run_ANN(ANN_input, weights_for_ANN);
                //cout << "agent output" << "\t" << agent.at(pop).agent_output << endl;
                get_error(pop);
            }
            //cout << "agent total error" << "\t" << agent.at(pop).error << endl;
            //cout << endl;
            agent.at(pop).age += 1;
            get_fitness(pop);
        }
        cout << endl;
        cout << endl;
        sort_indivduals_fitness();
        get_statistics();
        
        /*
         cout << "current population" << endl;
         for (int indv=0; indv<pP->pop_size; indv++)
         {
         cout << "agent" << "\t" << indv << endl;
         cout << "age" << "\t" << agent.at(indv).age << endl;
         cout << "fitness" << "\t" << agent.at(indv).fitness << endl;
         cout << endl;
         }
         */
        
        //cout << endl;
        if (gen < pP->max_gen-1)
        {
            natural_selection();
            
            /*
             cout << "new population" << endl;
             for (int indv=0; indv<pP->pop_size; indv++)
             {
             cout << "agent" << "\t" << indv << endl;
             cout << "output" << "\t" << agent.at(indv).fitness << endl;
             cout << endl;
             }
             
             for (int pop=0; pop<pP->pop_size; pop++)
             {
             cout << "agent" << "\t" <<  pop << "\t" << "weights" << endl;
             for (int n=0; n<pP->num_weights; n++)
             {
             cout << agent.at(pop).weights.at(n) << "\t";
             }
             cout << endl;
             }
             */
        }
    }
    write_statistics_to_file();
    get_best_agent();
}


#endif /* EA_hpp */
