from auxeticmop.sample_scripts import step1, step2, step3


"""
# Step 1
 Randomly generate parent topologies
 
# Step 2
Generate offspring from parent topologies

# Step 3
Evaluate fitness from parent and offspring results of 1st generation.
Select best topologies by pareto-front condition.
Export these as 2nd generation; "Topologies_2" and "FieldOutput_2".
"""

if __name__ == '__main__':
    step1.run()
    step2.run()
    step3.run()
