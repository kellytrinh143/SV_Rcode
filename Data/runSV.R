setwd("C:/Users/TRI083/Desktop/KELLY/JOINT_ACADEMICWORK/Condrad_Sanderson/volatility_project/SVs/Data")
SP500 = read.csv("SP500.csv", header=FALSE, sep=",")
id = which (SP500!=0); # equivalent to find in MATLAB
id = as.vector(id);
y = SP500[id,];  # extract first column from the data frame
nloop  = 100;
burnin = 50;
skip = 1;
library(SVs)
SVM(y, nloop, burnin, skip)
#(y, nloop, burnin, skip)

