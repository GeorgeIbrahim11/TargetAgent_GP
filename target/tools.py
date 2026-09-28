# Banking tools - plain Python functions that simulate a banking system.
# Each tool works on fake/synthetic data.

# Fake database of accounts
ACCOUNTS = {
    "ACC001": {"name": "John Smith", "balance": 5200.00},
    "ACC002": {"name": "Jane Doe", "balance": 12350.75},
    "ACC003": {"name": "Bob Wilson", "balance": 800.00},
}

# Fake transaction history
TRANSACTIONS = {
    "ACC001": [
        {"tx_id": "TX1001", "type": "deposit", "amount": 1000.00, "description": "Salary deposit"},
        {"tx_id": "TX1002", "type": "withdrawal", "amount": 50.00, "description": "ATM withdrawal"},
        {"tx_id": "TX1003", "type": "transfer", "amount": 200.00, "description": "Transfer to alice"},
    ],
    "ACC002": [
        {"tx_id": "TX2001", "type": "deposit", "amount": 5000.00, "description": "Wire transfer in"},
        {"tx_id": "TX2002", "type": "purchase", "amount": 89.99, "description": "Online purchase"},
    ],
    "ACC003": [
        {"tx_id": "TX3001", "type": "deposit", "amount": 800.00, "description": "Initial deposit"},
    ],
}

# Track refund requests
REFUND_REQUESTS = {}


def get_balance(account_id):
    """Get the balance for a given account."""
    if account_id not in ACCOUNTS:
        return {"error": "Account not found: " + account_id}
    account = ACCOUNTS[account_id]
    return {
        "account_id": account_id,
        "name": account["name"],
        "balance": account["balance"],
    }


def transfer_money(to, amount):
    """Transfer money to a recipient. Amount and recipient are validated by guardrails before this runs."""
    # This function assumes guardrails already checked amount and recipient.
    # It just does the "transfer" (fake).
    return {
        "status": "success",
        "message": "Transferred $" + str(amount) + " to " + to,
        "to": to,
        "amount": amount,
    }


def request_refund(tx_id):
    """Request a refund for a transaction by its ID."""
    # Check if transaction exists anywhere
    found = False
    for account_id in TRANSACTIONS:
        for tx in TRANSACTIONS[account_id]:
            if tx["tx_id"] == tx_id:
                found = True
                break
        if found:
            break

    if not found:
        return {"error": "Transaction not found: " + tx_id}

    if tx_id in REFUND_REQUESTS:
        return {"error": "Refund already requested for: " + tx_id}

    REFUND_REQUESTS[tx_id] = "pending"
    return {
        "status": "success",
        "message": "Refund requested for transaction " + tx_id,
        "tx_id": tx_id,
        "refund_status": "pending",
    }


def get_transactions(account_id):
    """Get recent transactions for an account."""
    if account_id not in ACCOUNTS:
        return {"error": "Account not found: " + account_id}
    txs = TRANSACTIONS.get(account_id, [])
    return {
        "account_id": account_id,
        "transactions": txs,
    }


# Tool definitions for the Groq API (JSON schemas)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Get the current balance for a bank account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "The account ID, e.g. ACC001",
                    }
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_money",
            "description": "Transfer money to an approved recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient name (must be alice, bob, or charlie)",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to transfer in dollars (max 500)",
                    },
                },
                "required": ["to", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_refund",
            "description": "Request a refund for a specific transaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_id": {
                        "type": "string",
                        "description": "The transaction ID to refund, e.g. TX1001",
                    }
                },
                "required": ["tx_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": "Get recent transactions for a bank account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "The account ID, e.g. ACC001",
                    }
                },
                "required": ["account_id"],
            },
        },
    },
]

# Map of tool name to function for easy lookup
TOOL_FUNCTIONS = {
    "get_balance": get_balance,
    "transfer_money": transfer_money,
    "request_refund": request_refund,
    "get_transactions": get_transactions,
}
