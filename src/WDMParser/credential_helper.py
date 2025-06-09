"""Helper functions for managing Google Cloud credentials."""

import os


def validate_credentials_path(credentials_path: str = None) -> str:
    """
    Validate and return the credentials path.
    
    Args:
        credentials_path: Path to the Google Cloud service account JSON key file.
                         If None, will try to get from CREDENTIALS_PATH environment variable,
                         then fallback to default locations.
    
    Returns:
        str: Valid credentials path
        
    Raises:
        ValueError: If no credentials path is provided
        FileNotFoundError: If the credentials file doesn't exist
    """
    if credentials_path is None:
        credentials_path = os.getenv("CREDENTIALS_PATH")
    
    # If no credentials_path provided and no env var, try default locations
    if not credentials_path:
        default_paths = [
            "key_vertex.json",
            "service-account-key.json", 
            "credentials.json",
            os.path.expanduser("~/key_vertex.json"),
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                credentials_path = path
                break
    
    if not credentials_path:
        raise ValueError(
            "No credentials path provided. Please either:\n"
            "1. Set CREDENTIALS_PATH environment variable: export CREDENTIALS_PATH='/path/to/key.json'\n"
            "2. Pass credentials_path parameter directly\n"
            "3. Place your JSON key file in the project root and name it 'key_vertex.json'"
        )
    
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"Credentials file not found at: {credentials_path}\n"
            "Please ensure:\n"
            "1. The path is correct\n"
            "2. The file exists and is accessible\n"
            "3. You have the proper Google Cloud service account JSON key"
        )
    
    return credentials_path


def setup_default_credentials() -> bool:
    """
    Try to setup default credentials if they don't exist.
    
    Returns:
        bool: True if credentials are available, False otherwise
    """
    # Check if already set
    if os.getenv("CREDENTIALS_PATH"):
        try:
            validate_credentials_path()
            return True
        except (ValueError, FileNotFoundError):
            pass
    
    # Try common default locations
    default_paths = [
        "key_vertex.json",
        "service-account-key.json",
        "credentials.json",
        os.path.expanduser("~/key_vertex.json"),
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            os.environ["CREDENTIALS_PATH"] = path
            print(f"Found credentials at: {path}")
            return True
    
    return False


def print_credentials_help():
    """Print helpful instructions for setting up credentials."""
    print("=" * 60)
    print("GOOGLE CLOUD CREDENTIALS SETUP")
    print("=" * 60)
    print()
    print("To use enrichment features, you need a Google Cloud service account key.")
    print()
    print("Steps to get your credentials:")
    print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
    print("2. Create a project or select an existing one")
    print("3. Enable the Vertex AI API")
    print("4. Go to IAM & Admin > Service Accounts")
    print("5. Create a new service account or select existing one")
    print("6. Download the JSON key file")
    print()
    print("Then set the environment variable:")
    print("export CREDENTIALS_PATH='/path/to/your/service-account-key.json'")
    print()
    print("Or place the file in your project root as 'key_vertex.json'")
    print("=" * 60) 