"""
Demo script showing the enhanced LLMAgentV2 capabilities.

This script demonstrates:
1. How to initialize the new enhanced agent
2. Enhanced reasoning formats
3. Comparison with the original agent
4. Analysis of reasoning patterns
"""

import json
from typing import List

from tau2.agent.llm_agent_completion_advanced import LLMAgentV2
from tau2.data_model.message import ToolMessage, UserMessage
from tau2.environment.tool import Tool, as_tool


# Example tools for demonstration
def get_user_info(user_id: str) -> str:
    """Get user information from the database."""
    users_db = {
        "123": {
            "name": "John Doe",
            "email": "john@example.com",
            "status": "active",
            "plan": "premium",
        },
        "456": {
            "name": "Jane Smith",
            "email": "jane@example.com",
            "status": "suspended",
            "plan": "basic",
        },
    }

    if user_id in users_db:
        return json.dumps(users_db[user_id])
    else:
        return json.dumps({"error": "User not found"})


def update_user_status(user_id: str, new_status: str) -> str:
    """Update user account status."""
    valid_statuses = ["active", "suspended", "pending"]
    if new_status not in valid_statuses:
        return json.dumps(
            {"error": f"Invalid status. Must be one of: {valid_statuses}"}
        )

    return json.dumps(
        {
            "user_id": user_id,
            "old_status": "suspended",
            "new_status": new_status,
            "updated": True,
            "timestamp": "2024-01-15T10:30:00Z",
        }
    )


def send_notification(
    user_id: str, message: str, notification_type: str = "email"
) -> str:
    """Send notification to user."""
    return json.dumps(
        {
            "user_id": user_id,
            "message": message,
            "type": notification_type,
            "sent": True,
            "notification_id": f"notif_{user_id}_{notification_type}",
        }
    )


# Create tools
tools = [
    as_tool(get_user_info),
    as_tool(update_user_status),
    as_tool(send_notification),
]

# Domain policy
DOMAIN_POLICY = """
Customer Service Policy:

1. Always verify user identity before making account changes
2. For account reactivation:
   - Check user information first
   - Update status to 'active' if request is valid
   - Send confirmation notification
   - Provide clear confirmation to user

3. Be helpful and professional at all times
4. If you cannot resolve an issue, escalate to human support
5. Always confirm successful actions with the user
"""


def demonstrate_enhanced_reasoning():
    """Demonstrate the enhanced reasoning capabilities."""

    # Initialize the enhanced agent
    agent = LLMAgentV2(
        tools=tools,
        domain_policy=DOMAIN_POLICY,
        llm="gpt-4",  # You'd use your actual model here
        enable_self_consistency=True,
    )

    print("=== LLMAgentV2 Enhanced Reasoning Demo ===\n")

    # Get initial state
    state = agent.get_init_state()

    # Simulate a complex user request
    user_message = UserMessage(
        role="user",
        content="Hi, my account (ID: 456) was suspended but I think it was a mistake. Can you please reactivate it and let me know when it's done?",
    )

    print("User Request:")
    print(f"'{user_message.content}'\n")

    print("System Prompt Structure:")
    print("✓ Role and Objective clearly defined")
    print(
        "✓ Core Agentic Principles (Persistence, Planning, Tool Mastery, Reflection, Verification)"
    )
    print("✓ Enhanced reasoning formats for complex tasks")
    print("✓ Error recovery protocols")
    print("✓ Self-consistency checking guidelines")
    print(
        "✓ Systematic workflow (Understand → Analyze → Plan → Execute → Reflect → Persist → Verify)"
    )
    print()

    # The actual LLM call would happen here, but for demo purposes,
    # let's show what the enhanced output format would look like:

    print("Expected Enhanced Reasoning Output Format:")
    print("""
Analysis:
The user is requesting account reactivation for ID 456. I need to:
1. Verify the user's current status
2. Understand why the account was suspended
3. Determine if reactivation is appropriate
4. Execute the reactivation process
5. Confirm completion to the user

Planning:
Step 1: Get user information for ID 456 to check current status
Step 2: If account is indeed suspended and reactivation is valid, update status to 'active'
Step 3: Send confirmation notification to the user
Step 4: Provide clear confirmation and next steps to the user

Execution Reasoning:
I'll start by retrieving the user information to understand the current account state and verify this is a legitimate reactivation request.

Verification:
This approach follows policy by verifying user info first, addresses the complete request systematically, and ensures proper notification. This is the appropriate first step.

Action:
{"name": "get_user_info", "arguments": {"user_id": "456"}}
""")

    # Demonstrate reasoning analysis
    print("\n=== Key Improvements Over Original Agent ===")
    print()
    print("1. **Multi-Step Reasoning Structure:**")
    print("   - Analysis: Problem decomposition")
    print("   - Planning: Step-by-step approach")
    print("   - Execution Reasoning: Action justification")
    print("   - Verification: Self-consistency check")
    print()

    print("2. **Agentic Workflow Principles:**")
    print("   - Persistence: Won't stop until issue is resolved")
    print("   - Planning: Plans next 2-3 steps ahead")
    print("   - Tool Mastery: Uses tools vs. guessing")
    print("   - Reflection: Learns from tool results")
    print()

    print("3. **Enhanced Error Handling:**")
    print("   - Acknowledges errors explicitly")
    print("   - Tries alternative strategies")
    print("   - Has clear escalation paths")
    print()

    print("4. **Context Management:**")
    print("   - Optimizes long conversations automatically")
    print("   - Preserves important tool results and reasoning")
    print("   - Tracks reasoning history for analysis")
    print()

    print("5. **Self-Consistency Features:**")
    print("   - Built-in accuracy checking")
    print("   - Policy compliance verification")
    print("   - Tool usage optimization")
    print()

    # Show reasoning analysis capabilities
    print("=== Reasoning Analysis Capabilities ===")

    # Simulate some reasoning history
    state.reasoning_history = [
        {
            "turn": 1,
            "reasoning": {
                "analysis": "User wants account reactivation for ID 456",
                "planning": "Check user info, update status, notify user",
                "execution": "Getting user information first",
                "verification": "Appropriate first step per policy",
            },
            "message_type": "tool_call",
        },
        {
            "turn": 2,
            "reasoning": {
                "analysis": "User account is suspended, reactivation seems legitimate",
                "planning": "Update status to active, send notification",
                "execution": "Updating account status",
                "verification": "Following policy requirements",
            },
            "message_type": "tool_call",
        },
    ]

    reasoning_summary = agent.get_reasoning_summary(state)
    print("\nReasoning Summary:")
    print(json.dumps(reasoning_summary, indent=2))


def show_comparison_with_original():
    """Show side-by-side comparison of original vs enhanced prompting."""

    print("\n=== Original vs Enhanced Prompting Comparison ===\n")

    print("ORIGINAL AGENT PROMPT:")
    print("─" * 50)
    print("""
Thought:
I need to help the user with their account reactivation.

Action:
{"name": "get_user_info", "arguments": {"user_id": "456"}}
""")

    print("\nENHANCED AGENT PROMPT:")
    print("─" * 50)
    print("""
Analysis:
The user is requesting account reactivation for ID 456. This requires:
1. Verifying current account status and suspension reason
2. Determining reactivation eligibility per policy
3. Executing reactivation process (status update + notification)
4. Confirming completion to user

Planning:
Step 1: Retrieve user info to verify current suspended status
Step 2: Update account status to 'active' if appropriate 
Step 3: Send confirmation notification to user
Step 4: Provide clear confirmation and any next steps

Execution Reasoning:
Starting with user info lookup follows policy requirement to verify before changes.
This gives me the context needed to proceed safely with reactivation.

Verification:
✓ Follows policy (verify first)
✓ Addresses complete user request
✓ Systematic approach reduces errors
✓ Tool usage is appropriate

Action:
{"name": "get_user_info", "arguments": {"user_id": "456"}}
""")

    print("\nKEY DIFFERENCES:")
    print("• Enhanced version breaks down the complete problem first")
    print("• Plans multiple steps ahead vs. single-step thinking")
    print("• Self-verifies reasoning for accuracy and policy compliance")
    print("• More systematic and less likely to miss requirements")
    print("• Provides transparency into AI decision-making process")


if __name__ == "__main__":
    demonstrate_enhanced_reasoning()
    show_comparison_with_original()

    print("\n=== How to Use LLMAgentV2 ===")
    print("""
# Basic Usage:
from tau2.agent.llm_agent_v2 import LLMAgentV2

agent = LLMAgentV2(
    tools=your_tools,
    domain_policy=your_policy,
    llm="your-model-name",
    enable_self_consistency=True,  # Enable enhanced reasoning
    max_context_tokens=100000      # Context window management
)

state = agent.get_init_state()
response, new_state = agent.generate_next_message(user_message, state)

# Analyze reasoning patterns:
reasoning_summary = agent.get_reasoning_summary(new_state)
print(f"Enhanced reasoning steps: {reasoning_summary['enhanced_reasoning_count']}")
""")

    print("\n✅ Demo completed! The enhanced agent is ready for testing.")
