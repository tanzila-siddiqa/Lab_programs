
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define Bayesian Network structure
model = DiscreteBayesianNetwork([('T', 'S'), ('S', 'F')])

# Step 2: Define Conditional Probability Distributions (CPDs)

# CPD for Temperature (T)
cpd_t = TabularCPD(
    variable='T',
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={'T': ['Normal', 'High']}
)

# CPD for Sweating (S)
cpd_s = TabularCPD(
    variable='S',
    variable_card=2,
    values=[[0.9, 0.4], [0.1, 0.6]],
    evidence=['T'],
    evidence_card=[2],
    state_names={'S': ['No', 'Yes'], 'T': ['Normal', 'High']}
)

# CPD for Fever (F)
cpd_f = TabularCPD(
    variable='F',
    variable_card=2,
    values=[[0.8, 0.2], [0.2, 0.8]],
    evidence=['S'],
    evidence_card=[2],
    state_names={'F': ['No', 'Yes'], 'S': ['No', 'Yes']}
)

# Step 3: Add CPDs to model and verify
model.add_cpds(cpd_t, cpd_s, cpd_f)

print("Model is valid:", model.check_model())

# Step 4: Perform inference
inference = VariableElimination(model)

query_result = inference.query(
    variables=['F'],
    evidence={'T': 'High'}
)

# Step 5: Display result
print("\nProbability of Fever given High Temperature:")
print(query_result)
