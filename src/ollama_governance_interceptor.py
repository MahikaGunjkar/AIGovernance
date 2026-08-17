import yaml
from agent_governance_toolkit_core.policy import PolicyEngine, PolicyEvaluator
from agent_governance_toolkit_core.audit import AuditLogger

class OllamaGovernanceInterceptor:
    def __init__(self, policy_path="policies/governance_policy.yaml"):
        self.policy_path = policy_path
        # Initialize AGT Core Policy Engine & Audit Logger
        self.engine = PolicyEngine.from_file(policy_path)
        self.logger = AuditLogger(log_file="heinzy_audit.log")

    def evaluate_tool_call(self, tool_name: str, tool_args: dict, user_query: str = "") -> dict:
        """
        Evaluates Ollama tool calling through Microsoft AGT Core modules.
        """
        action_type = tool_args.get("action_type", tool_name)
        url = tool_args.get("url", "")

        context = {
            "tool_name": tool_name,
            "action_type": action_type,
            "url": url,
            "query": user_query
        }

        # Evaluate policy via AGT PolicyEngine
        evaluation = self.engine.evaluate(context)
        return evaluation

    def log_activity(self, agent_id: str, user_query: str, tool_name: str, tool_args: dict, evaluation: dict, result_status: str, retrieved_chunks: list = None):
        """
        Logs complete activity lifecycle using AGT Audit Logger module.
        """
        self.logger.log_event(
            agent_id=agent_id,
            user_question=user_query,
            tool_accessed=tool_name,
            tool_arguments=tool_args,
            chunks_retrieved=retrieved_chunks or [],
            approval_decision=evaluation.get("decision"),
            matched_rule=evaluation.get("rule_matched"),
            status=result_status
        )

# Tool Execution Wrapper using AGT Modules
def execute_ollama_tool(tool_name: str, tool_args: dict, agent_id: str = "heinzy-advisor", query: str = ""):
    interceptor = OllamaGovernanceInterceptor()
    eval_result = interceptor.evaluate_tool_call(tool_name, tool_args, user_query=query)
    
    decision = eval_result.get("action", "DENY")
    
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
