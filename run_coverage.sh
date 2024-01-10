#!/bin/bash

###################################################################
#Script Name	: run_coverage                                                                                       
#Description	: This script runs tests and checks coverage                                                                       
#Args           : None                                                                                  
#Author         : Michael M. Jerge                                             
#Email          : mj6ux@virginia.edu                                    
###################################################################

# Run tests and get coverage
py.test --cov tests/
coverage report -m > coverage_report.txt
rm assets/coverage.svg
coverage-badge -o assets/coverage.svg -f
rm .coverage
rm test_all.py_log.jsonl