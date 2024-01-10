#!/bin/bash

###################################################################
#Script Name	: run_pylint                                                                                    
#Description	: This script runs Python linting and generates a 
#                 score
#Args           : None                                                                                  
#Author         : Michael M. Jerge                                           
#Email          : mj6ux@virginia.edu                                   
###################################################################

# Run linting and get score
pylint --rcfile=.pylintrc * | tee pylint_report.txt
score=$(sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p' pylint_report.txt)
echo "Pylint score was $score"
rm assets/pylint.svg
anybadge --value=$score --file=assets/pylint.svg pylint