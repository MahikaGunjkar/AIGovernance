import yaml
import json
from datetime import datetime
from urllib.parse import urlparse

class PolicyEngine:
    """
    Acts as the AGT Policy Decision Point (PDP).
    Evaluates context against governance rules.
    """
    def __init__(self, policy_path="policies/governance_policy.yaml"):
        self.policy_path = policy_path
        self.policy = self._load_policy()

    def _load_policy(self):
        try:
            with open(self.policy_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def evaluate(self, context: dict) -> dict:
        action_type = context.get("action_type")
        tool_name = context.get("tool_name")
        hostname = context.get("hostname", "")

        # 1. Block write operations
        if action_type in ["create", "update", "delete", "insert"]:
            return {
                "decision": "DENY",
                "rule": "block-write-actions",
                "reason": "Phase 1: Agent is strictly read-only."
            }

        # 2. Domain permissions checked by Policy Engine
        if action_type == "web_search" or tool_name == "web_search":
            if hostname == "cmu.edu" or hostname.endswith(".cmu.edu"):
                return {
                    "decision": "REQUIRE_APPROVAL",
                    "rule": "require-human-for-search",
                    "reason": "Human approval required for official CMU domains."
                }
            else:
                return {
                    "decision": "DENY",
                    "rule": "enforce-domain-allowlist",
                    "reason": "Web searches restricted to official university domains."
                }

        return {"decision": "ALLOW", "rule": "default-allow", "reason": "Action permitted."}

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
    """
    Acts as the Policy Enforcement Point (PEP).
    Parses facts and forwards context to the Policy Engine.
    """
    def __init__(self, policy_path="policies/governance_policy.yaml"):
        self.engine = PolicyEngine(policy_path)
        self.logger = AuditLogger(log_file="heinzy_audit.log")

    def evaluate_tool_call(self, tool_name: str, tool_args: dict, user_query: str = "") -> dict:
        # Step 1: Interceptor extracts facts ONLY (No verdict decision made here)
        raw_url = tool_args.get("url", "")
        parsed_hostname = urlparse(raw_url).netloc.lower() if raw_url else ""
        action_type = tool_args.get("action_type", tool_name)

        context = {
            "tool_name": tool_name,
            "action_type": action_type,
            "raw_url": raw_url,
            "hostname": parsed_hostname,
            "query": user_query
        }

        # Step 2: Delegate evaluation and verdict to AGT Policy Engine
        return self.engine.evaluate(context)

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
