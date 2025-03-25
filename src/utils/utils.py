import subprocess
import sys

def check_and_create_env(env_name: str = 'ford_case_iel') -> None:
    """
    Checks if the conda environment with the name `env_name` exists. 
    If not, it creates the environment using the `environment.yml` file.

    Parameters:
    env_name (str): The name of the environment to check/create. Default is 'ford_case'.
    """
    
    try:
        # Check if the conda environment exists
        result = subprocess.run(['conda', 'info', '--envs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # If the environment exists, return a message
        print(result)
        print(result.stdout.decode())
        if env_name in result.stdout.decode():
            print(f"Environment '{env_name}' already exists.")
        else:
            print(f"Environment '{env_name}' not found. Creating the environment...")
            # Create the environment from the environment.yml file
            subprocess.run(['conda', 'env', 'create', '-f', 'env.yml'], check=True)
            print(f"Environment '{env_name}' has been created.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while checking/creating the environment: {e}")
        sys.exit(1)

