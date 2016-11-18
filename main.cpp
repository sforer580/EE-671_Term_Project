//
//  main.cpp
//  EE-671_Term_Project
//
//  Created by Scott S Forer on 10/16/16.
//  Copyright © 2016 Scott S Forer. All rights reserved.
//

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "Parameters.hpp"
#include "ANN.hpp"
#include "EA.hpp"
#include "Individual.hpp"

int main()
{
    srand(time(NULL));
    Parameters P;
    Individual I;
    EA E;
    E.pP = &P;
    Neural_Network ANN;
    ANN.pP = &P;
    E.run_EA(I.weights);
    return 0;
}
