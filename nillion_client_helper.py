import os
import py_nillion_client as nillion
from nillion_payments_helper import create_payments_config


def create_nillion_client(userkey, nodekey):
    # Get the bootnode multiaddress from environment variables
    bootnode_multiaddress = os.getenv("NILLION_BOOTNODE_MULTIADDRESS")

    # Debug: Print the retrieved bootnode address
    print(f"Bootnode Multiaddress: {bootnode_multiaddress}")

    # Ensure the environment variable was retrieved successfully
    if not bootnode_multiaddress:
        raise ValueError("NILLION_BOOTNODE_MULTIADDRESS environment variable is not set or is empty.")

    bootnodes = [bootnode_multiaddress]
    payments_config = create_payments_config()

    return nillion.NillionClient(
        nodekey,
        bootnodes,
        nillion.ConnectionMode.relay(),
        userkey,
        payments_config  # Pass the payments_config as needed
    )
