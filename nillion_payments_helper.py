import os

class PaymentsConfig:
    def __init__(self, url, private_key, chain_id):
        self.url = url
        self.private_key = private_key
        self.chain_id = chain_id

    def validate(self):
        # Add validation logic if necessary
        if not self.url or not self.private_key or not self.chain_id:
            raise ValueError("Invalid PaymentsConfig")
        return True


def create_payments_config():
    """Create and return a PaymentsConfig object."""
    # Retrieve and validate environment variables
    rpc_endpoint = os.getenv("NILLION_NILCHAIN_JSON_RPC")
    private_key = os.getenv("NILLION_WALLET_PRIVATE_KEY")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")

    if None in (rpc_endpoint, private_key, chain_id):
        raise ValueError("One or more environment variables are missing.")

    # Return an instance of PaymentsConfig with the expected attributes
    return PaymentsConfig(
        url=rpc_endpoint,  # Use this as 'url'
        private_key=private_key,
        chain_id=chain_id
    )
