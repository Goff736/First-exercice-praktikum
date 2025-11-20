import pm4py
import pandas as pd

from pm4py.algo.evaluation.replay_fitness import algorithm as fitness
from pm4py.algo.evaluation.precision import algorithm as precision
from pm4py.algo.evaluation.generalization import algorithm as generalization
from pm4py.algo.evaluation.simplicity import algorithm as simplicity


file_path = r"C:\Users\SBS\Downloads\1st ex PRAK\BPI Challenge 2017_1_all\BPI Challenge 2017.xes.gz"
log = pm4py.read_xes(file_path)
df = pm4py.convert_to_dataframe(log)

print(df.head())
print(df.columns)

#Basic analysis
num_cases = df['case:concept:name'].nunique()
print("Number of cases:", num_cases)

num_events = len(df)
print("Number of events:", num_events)

variants = pm4py.algo.filtering.log.variants.variants_filter.get_variants(log)
num_variants = len(variants)
print("Number of process variants:", num_variants)

num_caselabels = df.filter(like='case:').shape[1]
num_eventlabels = df.shape[1] - num_caselabels
print("Number of case labels:", num_caselabels)
print("Number of event labels:", num_eventlabels)

events_case = df.groupby('case:concept:name').size()
mean_case_length = events_case.mean()
std_case_length = events_case.std()
print("Mean case length (events):", mean_case_length)
print("Std. dev. of case length:", std_case_length)

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
case_duration = df.groupby('case:concept:name')['time:timestamp'].agg(lambda x: x.max() - x.min())
mean_days = case_duration.mean().total_seconds() / (60 * 60 * 24)
std_days = case_duration.std().total_seconds() / (60 * 60 * 24)
print("Mean case duration (days):", mean_days)
print("Std. dev. of case duration (days):", std_days)

categorical_attributes = df.select_dtypes(include=['object']).columns
num_categorical_attributes = len(categorical_attributes)
print("Number of categorical event attributes:", num_categorical_attributes)

#optional
num_activities = df['concept:name'].nunique()
num_resources = df['org:resource'].nunique()
print("Number of unique activities:", num_activities)
print("Number of unique resources:", num_resources)

#aufgabe 3.3
print("algorithm1 :inductive miner")
try:
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.visualization.process_tree import visualizer as p_visualizer
    tree = inductive_miner.apply(log)
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    vis_tree = p_visualizer.apply(tree)
    p_visualizer.save(vis_tree,"process tree.png")
except Exception as e:
    print(f"error with inductive miner: {e}")

print("algo2: heuristics miner")
try:
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.visualization.petri_net import visualizer as n_visualizer

    netheu, intial_markingheu, final_markingheu = heuristics_miner.apply(log)
    vis_heu = n_visualizer.apply(netheu, intial_markingheu, final_markingheu)
    n_visualizer.save(vis_heu,"process heuristics tree.png")
except Exception as e:
    print(f"error with heuristics miner: {e}")

#BPMN model
print("create BPMN model")
try:
    from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
    parameters = {
        "format": "png",
        "zoom": 3.0
    }
    graph = pm4py.convert_to_bpmn(tree)
    vis_bpmn = bpmn_visualizer.apply(graph, parameters=parameters)
    bpmn_visualizer.save(vis_bpmn,"process_model.png")
    bpmn_visualizer.save(vis_bpmn, "process_model.bpmn")
except Exception as e:
    print(f"error with bpmn: {e}")

#evaluate inductive miner quality metrics
log_sample= pm4py.sample_cases(log,200)
fitness_in = fitness.apply(log_sample, net, initial_marking, final_marking)
precision_in = precision.apply(log_sample, net, initial_marking, final_marking)
generalization_in = generalization.apply(log_sample, net, initial_marking, final_marking)
simplicity_in = simplicity.apply(net)
print("\nInductive Miner Quality Metrics:")
print(f"  Fitness: {fitness_in['log_fitness']:.3f}")
print(f"  Precision: {precision_in:.3f}")
print(f"  Generalization: {generalization_in:.3f}")
print(f"  Simplicity: {simplicity_in:.3f}")






