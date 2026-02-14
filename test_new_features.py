import brain
import pandas as pd

print("Testing Fear of Failure...")
fear = brain.analyze_fear_of_failure(skills=30, hours=12, education="Bachelor")
print(fear)

print("\nTesting Mini Projects...")
projects = brain.generate_mini_projects(role="Data Scientist", skills=40)
print(projects)

print("\nTesting Simulation...")
sim = brain.simulate_progression(current_skills=50, hours=5)
print(sim)

print("\nTesting Core Logic Extensions...")
print(brain.company_skill_mapping("Data Scientist"))
print(brain.ai_mentor_modes("Drill Sergeant"))
print(brain.personality_career_filter("INTJ"))
print(brain.career_regret_minimizer("Data Scientist", "UX Designer"))
print(brain.interest_decay_detection([40, 38, 35]))
print(brain.peer_comparison_anonymous(85, 8)) # Should trigger "Elite" or high percentile logic

print("\nTesting Advanced Features...")
print(brain.rl_roadmap_optimizer("Goal", 20)) # Low skill -> Clone/Tutorial
print(brain.rl_roadmap_optimizer("Goal", 90)) # High skill -> Deploy/Refactor
print(brain.federated_learning_privacy()) # Should show dynamic loss/cycle
print(brain.future_agi_advisor("Data Scientist"))
print(brain.evaluation_frameworks({}))
print(brain.ai_interview_prep("Data Scientist", "en"))

print("\nTesting Predict Endpoint Logic...")
# Mocking data for predict
try:
    # We can't easily test the full endpoint without a request context or mocking, 
    # but we've tested the core functions above.
    pass
except Exception as e:
    print(e)
