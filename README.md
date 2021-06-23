# NumeralClassification

In order to perform training with the parameters that were used as a benchmark for the model, use the following command line argument:

python NumeralClassification.py -F mnist_test.csv -A 0.99 -E 300

If you wish to perform training with custom parameters, use the list below to determine what arguments are available to use:

-F (String, Required):  The filename (and path) of the CSV file that contains the data
-S (int, Optional):     The maximum number of samples the program will take from the data file
-A (float, Optional):   The accuracy level to which we wish to train the model
-E (int, Optional):     The number of epochs for which we wish to train the model
-T (float, Optional):   The percentage of the data set we wish to allocate to testing (the rest is used for training)

All optional arguments have default values:

-S (int, Optional):     10000
-A (float, Optional):   0.85
-E (int, Optional):     100
-T (float, Optional):   0.5
