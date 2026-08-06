import json
from datetime import datetime
from agent_governance import govern, PolicyEnforcer

enforcer = PolicyEnforcer(policy_path="../policies/governance_policy.yaml")

@govern(enforcer)
def execute_agent_action(action_type: str, url: str = None, data: dict = None):
    print(f"Executing {action_type}...")
    return {"status": "success", "action": action_type}

def log_event(advisor_id, query, policy_eval_result):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "advisor_id": advisor_id,
        "query": query,
        "policy_evaluation": policy_eval_result
    }
    with open("heinzy_audit.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
