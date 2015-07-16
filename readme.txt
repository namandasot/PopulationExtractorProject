Step 1 Classifier:
Used to classify whether given doc contains population abstract or not.

use : python filename TrainFileForPositiveCase TrainFileForNegativeCase TestFileForPositiveCase TestFileForNegativeCase
Example:  python newBayes.py trainClass1.txt trainClass0.txt testClass1.txt testClass0.txt
 
 (here we are taking two input files because the code was used for testing and analysing purposes and it was easy to analyse data if poitive and negative cases are separated) 
 
 this will generate 2 files. out1.txt and out0.txt
 
 Step 2 Extractor:
 Place above output file along with corresponding testClass file in DataSet folder of Step2Extractor folder. 
 Run the populationExtr.java and extract Population phrases. Result will also be stored in DataSet folder as ResultFile.txt .
 Dependencies are : stanford-parser.jar and  stanford-parser-3.4-models.jar
 
 Add Dependencies/Jar files from the Dependency folder.
 (stanford-parser-3.4-models.jar  is too big. Available at " https://drive.google.com/file/d/0B_wjmbf3y6wUcnNUS0JmdHRiVkU/view?usp=sharing " )
 
