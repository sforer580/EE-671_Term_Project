//
//  ANN.hpp
//  EE-671_Term_Project
//
//  Created by Scott S Forer on 10/16/16.
//  Copyright © 2016 Scott S Forer. All rights reserved.
//

#ifndef ANN_hpp
#define ANN_hpp

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <iomanip>



using namespace std;


class Node
{
    friend class Layer;
    
public:
    double element;
    
};


class Layer
{
    friend class Neural_Network;
    friend class Node;
    
public:
    vector<Node> neuron;
    
};


class Neural_Network
{
    friend class Layer;
    friend class EA;
    
public:
    Parameters* pP;
    vector<Layer> lay;
    
    void build_ANN();
    
    double state;
    void communication(double ANN_input, vector<double> weights_for_ANN);
    vector<double> i_h_w;
    void get_i_h_w(vector<double> weights_for_ANN);
    vector<double> h_o_w;
    void get_h_o_w(vector<double> weights_for_ANN);
    void normalize_inputs();
    
    void get_inputs();
    
    void build_input_nodes();
    
    vector<double> input_to_hidden_layer_connections;
    void build_input_to_hidden_layer_connection();
    void sum_input_to_hidden_layer_connections();
    
    vector<double> hidden_to_output_layer_connections;
    void build_hidden_to_output_layer_connection();
    void sum_hidden_to_output_layer_connections();
    
    double sigmoid_function(double sigmoid_input, double sigmoid_output);
    
    void scale_outputs();
    double output;
    
    double run_ANN(double ANN_input, vector<double> weights_for_ANN);
    
    
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ANN Build Functions


////////////////////////////////////////////////////////////////
//Builds the instances for the ANN
void Neural_Network::build_ANN()
{
    //cout << pP->num_layers << endl;
    //cout << pP->input_layer_size << endl;
    //cout << pP->output_layer_size << endl;
    for (int l=0; l<pP->num_layers; l++)
    {
        Layer L;
        lay.push_back(L);
        if (l==0)
        {
            for (int n=0; n<pP->input_layer_size+1; n++)
            {
                Node N;
                lay.at(l).neuron.push_back(N);
            }
        }
        if (l==1)
        {
            for (int n=0; n<pP->hidden_layer_size+1; n++)
            {
                Node N;
                lay.at(l).neuron.push_back(N);
            }
        }
        if (l==2)
        {
            for (int n=0; n<pP->output_layer_size; n++)
            {
                Node N;
                lay.at(l).neuron.push_back(N);
            }
        }
    }
}


////////////////////////////////////////////////////////////////
//Assigns the inputs and weights to Neural Network class variables
void Neural_Network::normalize_inputs()
{
    state = (state - pP->input_lower_limit)/(pP->input_upper_limit - pP->input_lower_limit);
}


////////////////////////////////////////////////////////////////
//gets the input to hidden layer weights
void Neural_Network::get_i_h_w(vector<double> weights_for_ANN)
{
    i_h_w.resize(pP->num_i_h_w);
    for (int w=0; w<pP->num_i_h_w; w++)
    {
        i_h_w.at(w) = weights_for_ANN.at(w);
    }
    /*
     cout << "input to hidden layer weights" << endl;
     for (int w=0; w<pP->num_i_h_w; w++)
     {
     cout << i_h_w.at(w) << "\t";
     }
     cout << endl;
     cout << endl;
     */
}


////////////////////////////////////////////////////////////////
//gets the hidden to output layer weights
void Neural_Network::get_h_o_w(vector<double> weights_for_ANN)
{
    h_o_w.resize(pP->num_h_o_w);
    for (int w=0; w<pP->num_h_o_w; w++)
    {
        h_o_w.at(w) = weights_for_ANN.at(w+pP->num_i_h_w);
    }
    /*
     cout << "hidden to output layer weights" << endl;
     for (int w=0; w<pP->num_h_o_w; w++)
     {
     cout << h_o_w.at(w) << "\t";
     }
     cout << endl;
     cout << endl;
     */
}



////////////////////////////////////////////////////////////////
//Get Communication
//Assigns the inputs and weights to Neural Network class variables
void Neural_Network::communication(double ANN_input, vector<double> weights_for_ANN)
{
    state = ANN_input;
    //cout << "input" << "\t" << state << endl;
    //cout << endl;
    normalize_inputs();
    get_i_h_w(weights_for_ANN);
    get_h_o_w(weights_for_ANN);
    
}


////////////////////////////////////////////////////////////////
//Gets the inputs for the input layer
void Neural_Network::get_inputs()
{
    lay.at(0).neuron.at(0).element = state;
    //cout << "normalized input" << "\t" << lay.at(0).neuron.at(0).element << endl;
    //cout << endl;
}


////////////////////////////////////////////////////////////////
//Build input noded
void Neural_Network::build_input_nodes()
{
    for (int n=0; n<lay.at(0).neuron.size(); n++)
    {
        if (n<lay.at(0).neuron.size()-1)
        {
            lay.at(0).neuron.at(n).element = state;
        }
        if (n==lay.at(0).neuron.size()-1)
        {
            lay.at(0).neuron.at(n).element = 1;
        }
    }
}




////////////////////////////////////////////////////////////////
//Builds the input to hidden layer connections
void Neural_Network::build_input_to_hidden_layer_connection()
{
    //cout << lay.at(1).neuron.size()-1 << endl;
    input_to_hidden_layer_connections.resize(lay.at(0).neuron.size()*(lay.at(1).neuron.size()-1));
    for (int l0=0; l0<lay.at(0).neuron.size(); l0++)
    {
        if (l0<lay.at(0).neuron.size()-1)
        {
            //cout << lay.at(1).neuron.size() << endl;
            for (int l1=0; l1<lay.at(1).neuron.size()-1; l1++)
            {
                input_to_hidden_layer_connections.at(l0+l1) = state*i_h_w.at(l0);
            }
        }
        if (l0 == lay.at(0).neuron.size()-1)
        {
            for (int l1=0; l1<lay.at(1).neuron.size()-1; l1++)
            {
                input_to_hidden_layer_connections.at(l0+pP->hidden_layer_size-1+l1) = lay.at(0).neuron.at(l0).element*i_h_w.at(l0);
            }
        }
    }
    
    /*
     cout << "input to hidden layer connections before weights" << endl;
     for (int i=0; i<input_to_hidden_layer_connections.size(); i++)
     {
     cout << input_to_hidden_layer_connections.at(i) << "\t";
     }
     cout << endl;
     
     for (int c=0; c<input_to_hidden_layer_connections.size(); c++)
     {
     input_to_hidden_layer_connections.at(c) = input_to_hidden_layer_connections.at(c)*i_h_w.at(c);
     }
     
     cout << "input to hidden layer connections after weights" << endl;
     for (int i=0; i<input_to_hidden_layer_connections.size(); i++)
     {
     cout << input_to_hidden_layer_connections.at(i) << "\t";
     }
     cout << endl;
     cout << endl;
     */
}


////////////////////////////////////////////////////////////////
//Get sum for hidden layer
void Neural_Network::sum_input_to_hidden_layer_connections()
{
    for (int n=0; n<lay.at(1).neuron.size()-1; n++)
    {
        double sum= 0;
        for (int c=0; c<input_to_hidden_layer_connections.size()/(lay.at(1).neuron.size() -1); c++)
        {
            sum += input_to_hidden_layer_connections.at(n+c*(lay.at(1).neuron.size()-1));
        }
        lay.at(1).neuron.at(n).element = sum;
    }
    lay.at(1).neuron.at(lay.at(1).neuron.size()-1).element = 1;
    
    for (int n=0; n<lay.at(1).neuron.size()-1; n++)
    {
        double sigmoid_input = 0;
        double sigmoid_output = 0;
        sigmoid_input = lay.at(1).neuron.at(n).element;
        sigmoid_output = sigmoid_function(sigmoid_input, sigmoid_output);
        lay.at(1).neuron.at(n).element = sigmoid_output;
    }
    
}


////////////////////////////////////////////////////////////////
//Builds the hidden to output layer connections
void Neural_Network::build_hidden_to_output_layer_connection()
{
    hidden_to_output_layer_connections.resize(lay.at(1).neuron.size()*lay.at(2).neuron.size());
    for (int l1=0; l1<lay.at(1).neuron.size(); l1++)
    {
        if (l1<lay.at(1).neuron.size()-1)
        {
            for (int l2=0; l2<lay.at(2).neuron.size(); l2++)
            {
                hidden_to_output_layer_connections.at(l1) = lay.at(1).neuron.at(l1).element*h_o_w.at(l1);
            }
        }
        if (l1 == lay.at(1).neuron.size()-1)
        {
            for (int l2=0; l2<lay.at(2).neuron.size(); l2++)
            {
                hidden_to_output_layer_connections.at(l1+l2) = 1*h_o_w.at(l1);
            }
        }
    }
    /*
     cout << "hidden to output layer connections before weights" << endl;
     for (int i=0; i<hidden_to_output_layer_connections.size(); i++)
     {
     cout << hidden_to_output_layer_connections.at(i) << "\t";
     }
     cout << endl;
     
     for (int c=0; c<hidden_to_output_layer_connections.size(); c++)
     {
     hidden_to_output_layer_connections.at(c) = hidden_to_output_layer_connections.at(c)*h_o_w.at(c);
     }
     cout << "hidden to output layer connections after weights" << endl;
     for (int i=0; i<hidden_to_output_layer_connections.size(); i++)
     {
     cout << hidden_to_output_layer_connections.at(i) << "\t";
     }
     cout << endl;
     */
}

////////////////////////////////////////////////////////////////
//Get value for output layer nodes
void Neural_Network::sum_hidden_to_output_layer_connections()
{
    for (int n=0; n<lay.at(2).neuron.size(); n++)
    {
        double sum= 0;
        for (int c=0; c<hidden_to_output_layer_connections.size()/(lay.at(2).neuron.size()); c++)
        {
            sum += hidden_to_output_layer_connections.at(n+c*(lay.at(2).neuron.size()));
        }
        lay.at(2).neuron.at(n).element = sum;
    }
    
    for (int n=0; n<lay.at(2).neuron.size(); n++)
    {
        double sigmoid_input = 0;
        double sigmoid_output = 0;
        sigmoid_input = lay.at(2).neuron.at(n).element;
        sigmoid_output = sigmoid_function(sigmoid_input, sigmoid_output);
        lay.at(2).neuron.at(n).element = sigmoid_output;
    }
    
}


////////////////////////////////////////////////////////////////
//runs the sigmoid function
double Neural_Network::sigmoid_function(double sigmoid_input, double sigmoid_output)
{
    //sigmoid_output = 1/(1+exp(-sigmoid_input));
    sigmoid_output = tanh(sigmoid_input);
    return(sigmoid_output);
}


////////////////////////////////////////////////////////////////
//Sacle outputs
void Neural_Network::scale_outputs()
{
    output = 0;
    output = (lay.at(2).neuron.at(0).element*(pP->output_upper_limit - pP->output_lower_limit)) + pP->output_lower_limit;
    //cout << "ANN output" << "\t" << output << endl;
}




////////////////////////////////////////////////////////////////
//Gets the inputs for the input layer
double Neural_Network::run_ANN(double ANN_input, vector<double> weights_for_ANN)
{
    communication(ANN_input, weights_for_ANN);
    get_inputs();
    build_input_nodes();
    build_input_to_hidden_layer_connection();
    sum_input_to_hidden_layer_connections();
    build_hidden_to_output_layer_connection();
    sum_hidden_to_output_layer_connections();
    /*
     cout << endl;
     
     for (int l=0; l<3; l++)
     {
     cout << "Layer" << "\t" << l << endl;
     for (int n=0; n<lay.at(l).neuron.size(); n++)
     {
     cout << "node" << "\t" << n << endl;
     cout << "value" << "\t" << lay.at(l).neuron.at(n).element << endl;
     }
     cout << endl;
     cout << endl;
     }
     cout << endl;
     */
    scale_outputs();
    //cout << "ANN output" << "\t" << output << endl;
    return output;
}








#endif /* ANN_hpp */
