import yaml
import json
from datetime import datetime

class OllamaGovernanceInterceptor:
    def __init__(self, policy_path="policies/governance_policy.yaml"):
        with open(policy_path, "r") as f:
            self.policy = yaml.safe_load(f)
        self.rules = self.policy.get("rules", [])

    def evaluate_tool_call(self, tool_name: str, tool_args: dict) -> dict:
        """
        Intercepts Ollama tool calling before execution.
        Returns evaluation decision: ALLOW, DENY, or REQUIRE_APPROVAL.
        """
        action_type = tool_args.get("action_type", tool_name)
        url = tool_args.get("url", "")

        # 1. Check write action restrictions
        if action_type in ["create", "update", "delete", "insert"]:
            return self._format_decision("DENY", "block-write-actions", "Phase 1: Agent is strictly read-only.")

        # 2. Check web search domain restrictions
        if action_type == "web_search" or tool_name == "web_search":
            if not url.endswith(".cmu.edu"):
                return self._format_decision("DENY", "enforce-domain-allowlist", "Phase 2: Web searches restricted to official university domains.")
            else:
                return self._format_decision("REQUIRE_APPROVAL", "require-human-for-search", "Human approval required for cmu.edu searches.")

        return self._format_decision("ALLOW", "default-allow", "Action permitted.")

    def log_audit_event(self, agent_id: str, query: str, decision: dict):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent_id": agent_id,
            "query": query,
            "policy_evaluation": {
                "action_denied": decision["action"] == "DENY",
                "hitl_triggered": decision["action"] == "REQUIRE_APPROVAL",
                "rule_matched": decision["rule"]
            }
        }
        with open("heinzy_audit.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _format_decision(self, action, rule, description):
        return {
            "action": action,
            "rule": rule,
            "description": description
        }

# Ollama Tool Call Wrapper Function
def execute_ollama_tool(tool_name: str, tool_args: dict, agent_id: str = "heinzy-advisor", query: str = ""):
    interceptor = OllamaGovernanceInterceptor()
    decision = interceptor.evaluate_tool_call(tool_name, tool_args)
    interceptor.log_audit_event(agent_id, query, decision)

    if decision["action"] == "DENY":
        raise PermissionError(f"[GOVERNANCE DENIED] Rule: {decision['rule']} - {decision['description']}")
    elif decision["action"] == "REQUIRE_APPROVAL":
        print(f"[GOVERNANCE PAUSED] Approval needed for tool '{tool_name}'.")
        return {"status": "PAUSED_FOR_APPROVAL", "details": decision}

    print(f"[GOVERNANCE PASSED] Executing tool '{tool_name}'...")
    # Proceed to execute tool function here
    return {"status": "SUCCESS", "details": decision}
