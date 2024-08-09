import os
import sys
import asyncio
import pandas as pd
import nada_numpy as na
import nada_numpy.client as na_client
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import create_nillion_client
from nillion_payments_helper import create_payments_config
from sklearn.linear_model import LinearRegression
from nada_ai.client import SklearnClient
from nillion_utils import compute, store_program, store_secrets, get_user_id_by_seed
import py_nillion_client as nillion

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Set pandas option to retain old downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

def check_environment_variables():
    required_vars = [
        "NILLION_CLUSTER_ID",
        "NILLION_NILCHAIN_GRPC",
        "NILLION_NILCHAIN_CHAIN_ID",
        "NILLION_NILCHAIN_PRIVATE_KEY_0"
    ]
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Environment variable {var} is missing or empty.")
        print(f"Environment variable {var} is set.")

def transform_and_save_data(input_file, output_file):
    """Transform the input CSV file and save it with the required format."""
    print(f"Transforming data from {input_file} to {output_file}")
    data = pd.read_csv(input_file)
    data = data.replace({'yes': 1, 'no': 0}).infer_objects(copy=False)
    data.to_csv(output_file, index=False)
    print(f"Data transformation complete. Saved to {output_file}")
    return output_file

async def main():
    try:
        print("Starting HealthGuard AI main script...")

        # Check environment variables
        check_environment_variables()
        print("Loaded environment variables:")
        print(f"NILLION_CLUSTER_ID: {os.getenv('NILLION_CLUSTER_ID')}")
        print(f"NILLION_NILCHAIN_GRPC: {os.getenv('NILLION_NILCHAIN_GRPC')}")
        print(f"NILLION_NILCHAIN_CHAIN_ID: {os.getenv('NILLION_NILCHAIN_CHAIN_ID')}")
        print(f"NILLION_NILCHAIN_PRIVATE_KEY_0: {os.getenv('NILLION_NILCHAIN_PRIVATE_KEY_0')}")

        # Set fixed-point number scaling
        na.set_log_scale(32)
        print("Fixed-point number scaling set.")

        # Define program name and path to compiled program binary
        program_name = "train_and_export_model"
        program_mir_path = f"./target/{program_name}.nada.bin"
        print(f"Program path: {program_mir_path}")

        # Check if the program file exists
        if not os.path.exists(program_mir_path):
            raise FileNotFoundError(f"The program file {program_mir_path} does not exist.")
        print("Program file exists.")

        # Initialize parties
        party_names = na_client.parties(2)
        seed_0, seed_1 = 'seed-party-model', 'seed-party-input'
        print(f"Initialized parties: {party_names}")

        # Create Nillion Clients
        print("Creating Nillion Clients...")
        client_0 = create_nillion_client(nillion.UserKey.from_seed(seed_0), nillion.NodeKey.from_seed(seed_0))
        client_1 = create_nillion_client(nillion.UserKey.from_seed(seed_1), nillion.NodeKey.from_seed(seed_1))
        print("Nillion Clients created.")

        # Retrieve party and user IDs
        party_id_0, user_id_0 = client_0.party_id, client_0.user_id
        party_id_1, user_id_1 = client_1.party_id, client_1.user_id
        print(f"Party and user IDs retrieved: party_id_0={party_id_0}, user_id_0={user_id_0}, party_id_1={party_id_1}, user_id_1={user_id_1}")

        # Configure payments
        print("Configuring payments...")
        payments_config = create_payments_config()
        payments_client = LedgerClient(payments_config)
        payments_wallet = LocalWallet(PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))), prefix="nillion")
        print("Payments configured.")

        # Store the compiled Nada program on the Nillion network
        print("Storing the compiled Nada program...")
        program_id = await store_program(client_0, payments_wallet, payments_client, user_id_0, os.getenv('NILLION_CLUSTER_ID'), program_name, program_mir_path)
        print(f"Program stored with ID: {program_id}")

        # Load and preprocess data
        input_file = 'data/diabetes.csv'
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"The input file {input_file} does not exist.")
        print(f"Input file {input_file} exists.")

        data_file = transform_and_save_data(input_file, './diabetes-transformed.csv')
        data = pd.read_csv(data_file)
        features = [col for col in data.columns if col != 'Outcome']
        X = data[features].values
        y = data['Outcome'].values
        print(f"Data loaded and preprocessed. Features: {features}")

        # Convert data to Nada arrays
        X_nada = [na.array(row, "X_data", na.SecretRational) for row in X]
        y_nada = na.array(y, "y_data", na.SecretRational)

        # Train a Linear Regression model
        print("Training Linear Regression model...")
        model = LinearRegression().fit(X, y)
        print("Learned model coeffs:", model.coef_)
        print("Learned model intercept:", model.intercept_)

        # Create and store model secrets on Nillion
        print("Creating and storing model secrets...")
        model_client = SklearnClient(model)
        model_secrets = nillion.NadaValues(model_client.export_state_as_secrets("my_model", na.SecretRational))

        # Set permissions and allow computation for specific user IDs
        print("Setting permissions...")
        permissions = nillion.Permissions.default_for_user(user_id_0)
        allowed_user_ids = [user_id_1, get_user_id_by_seed("inference_1"), get_user_id_by_seed("inference_2"),
                            get_user_id_by_seed("inference_3")]
        permissions.add_compute_permissions({user: {program_id} for user in allowed_user_ids})
        model_store_id = await store_secrets(client_0, payments_wallet, payments_client, os.getenv('NILLION_CLUSTER_ID'), model_secrets, 1, permissions)
        print(f"Model secrets stored with ID: {model_store_id}")

        # Prepare new input data
        print("Preparing new input data...")
        input_data = na.array([2, 120, 80, 20, 85, 30.5, 0.627, 45], "my_input", na.SecretRational)
        print(f"Input data prepared: {input_data}")

        # Set up compute bindings for the parties
        print("Setting up compute bindings...")
        compute_bindings = nillion.ProgramBindings(program_id)
        compute_bindings.add_input_party(party_names[0], party_id_0)
        compute_bindings.add_input_party(party_names[1], party_id_1)
        compute_bindings.add_output_party(party_names[1], party_id_1)
        print("Compute bindings set up.")

        print(f"Computing using program {program_id}")
        print(f"Use secret store_id: {model_store_id}")

        # Perform the blind computation
        print("Performing blind computation...")
        inference_result = await compute(client_1, payments_wallet, payments_client, program_id, os.getenv('NILLION_CLUSTER_ID'), compute_bindings, [model_store_id], input_data, verbose=True)
        print("Blind computation completed.")

        # Process the output
        output = na_client.float_from_rational(inference_result["my_output"])
        expected = model.predict(np.array([2, 120, 80, 20, 85, 30.5, 0.627, 45]).reshape(1, -1))[0]
        print(f"🙈 Result computed by Nada program: {output}")
        print(f"👀 Expected result computed by sklearn: {expected}")

        # Print the input data and predicted outcome
        print("\nFeatures of new patient:")
        for feature, value in zip(features, [2, 120, 80, 20, 85, 30.5, 0.627, 45]):
            print(f"    {feature}: {value}")
        print(f"\nPredicted diabetes outcome: {output}")

        print("Script completed successfully.")
        return inference_result

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(main())
    if result is None:
        print("The script encountered an error and could not complete successfully.")
    else:
        print("The script completed successfully.")
