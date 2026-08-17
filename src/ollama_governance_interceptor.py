import yaml
import json
from datetime import datetime
from urllib.parse import urlparse

class PolicyEngine:
    def __init__(self, policy_path="policies/governance_policy.yaml"):
        self.policy_path = policy_path
        try:
            with open(policy_path, "r") as f:
                self.policy = yaml.safe_load(f) or {}
        except Exception:
            self.policy = {}

    def evaluate(self, tool_name: str, tool_args: dict) -> dict:
        action_type = tool_args.get("action_type", tool_name)
        url = tool_args.get("url", "")

        # 1. Deny write/mutation actions
        if action_type in ["create", "update", "delete", "insert"]:
            return {
                "decision": "DENY",
                "action": "DENY",
                "rule": "block-write-actions",
                "reason": "Phase 1: Agent is strictly read-only."
            }

        # 2. Check web search domain permissions
        if action_type == "web_search" or tool_name == "web_search":
            domain = urlparse(url).netloc.lower()
            if domain == "cmu.edu" or domain.endswith(".cmu.edu"):
                return {
                    "decision": "REQUIRE_APPROVAL",
                    "action": "REQUIRE_APPROVAL",
                    "rule": "require-human-for-search",
                    "reason": "Human approval required for official CMU domains."
                }
            else:
                return {
                    "decision": "DENY",
                    "action": "DENY",
                    "rule": "enforce-domain-allowlist",
                    "reason": "Phase 2: Web searches restricted to official university domains."
                }

        return {
            "decision": "ALLOW",
            "action": "ALLOW",
            "rule": "default-allow",
            "reason": "Action permitted."
        }

class AuditLogger:
    def __init__(self, log_file="heinzy_audit.log"):
        self.log_file = log_file

    def log_event(self, agent_id: str, query: str, tool_name: str, tool_args: dict, evaluation: dict, status: str):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent_id": agent_id,
            "query": query,
            "tool_accessed": tool_name,
            "tool_arguments": tool_args,
            "policy_evaluation": {
                "action_denied": evaluation.get("decision") == "DENY",
                "hitl_triggered": evaluation.get("decision") == "REQUIRE_APPROVAL",
                "rule_matched": evaluation.get("rule"),
                "reason": evaluation.get("reason")
            },
            "status": status
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

class OllamaGovernanceInterceptor:
    def __init__(self, policy_path="policies/governance_policy.yaml"):
        self.engine = PolicyEngine(policy_path)
        self.logger = AuditLogger(log_file="heinzy_audit.log")

    def evaluate_tool_call(self, tool_name: str, tool_args: dict, user_query: str = "") -> dict:
        return self.engine.evaluate(tool_name, tool_args)

    def log_activity(self, agent_id: str, user_query: str, tool_name: str, tool_args: dict, evaluation: dict, status: str):
        self.logger.log_event(agent_id, user_query, tool_name, tool_args, evaluation, status)

def execute_ollama_tool(tool_name: str, tool_args: dict, agent_id: str = "heinzy-advisor", query: str = ""):
    interceptor = OllamaGovernanceInterceptor()
    eval_result = interceptor.evaluate_tool_call(tool_name, tool_args, user_query=query)
    decision = eval_result.get("decision", "DENY")
    
    if decision == "DENY":
        interceptor.log_activity(agent_id, query, tool_name, tool_args, eval_result, "DENIED")
        raise PermissionError(f"[AGT DENIED] Rule: {eval_result.get('rule')} - {eval_result.get('reason')}")
        
    elif decision == "REQUIRE_APPROVAL":
        interceptor.log_activity(agent_id, query, tool_name, tool_args, eval_result, "PAUSED_FOR_APPROVAL")
        print(f"[AGT PAUSED] Approval required for tool '{tool_name}'.")
        return {"status": "PAUSED_FOR_APPROVAL", "details": eval_result}

    interceptor.log_activity(agent_id, query, tool_name, tool_args, eval_result, "SUCCESS")
    print(f"[AGT PASSED] Executing tool '{tool_name}'...")
    return {"status": "SUCCESS", "details": eval_result}
