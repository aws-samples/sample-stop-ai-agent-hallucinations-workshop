"""
Neo4j Configuration Helper for Workshop Studio

This module helps load Neo4j credentials in Jupyter notebooks.
It tries multiple sources in order:
1. Environment variables (set by Workshop Studio)
2. AWS Secrets Manager (fallback for local testing)

Usage in your notebook:
    from neo4j_config import get_neo4j_credentials

    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD = get_neo4j_credentials()

    # Use with neo4j driver
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
"""

import os


def get_neo4j_credentials():
    """
    Get Neo4j connection credentials from environment or Secrets Manager.

    Returns:
        tuple: (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    Raises:
        ValueError: If credentials cannot be found
    """
    # Try environment variables first (Workshop Studio sets these)
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    # If not in environment, try to get from CloudFormation/Secrets Manager
    # (useful for local testing with your own stack)
    if not neo4j_uri or not neo4j_password:
        try:
            import boto3

            # Try to get from CloudFormation stack outputs
            # Note: This requires AWS credentials and the stack name
            # In Workshop Studio, environment variables are pre-configured
            cf = boto3.client('cloudformation')

            # Look for a workshop stack (you may need to adjust the stack name)
            stack_name = os.getenv('WORKSHOP_STACK_NAME', 'test-workshop-code-editor')

            try:
                response = cf.describe_stacks(StackName=stack_name)
                outputs = {o['OutputKey']: o['OutputValue']
                          for o in response['Stacks'][0]['Outputs']}

                if 'Neo4jEndpoint' in outputs:
                    neo4j_host = outputs['Neo4jEndpoint']
                    neo4j_uri = f"bolt://{neo4j_host}:7687"

                if 'Neo4jPasswordSecretArn' in outputs and not neo4j_password:
                    sm = boto3.client('secretsmanager')
                    secret_arn = outputs['Neo4jPasswordSecretArn']
                    neo4j_password = sm.get_secret_value(SecretId=secret_arn)['SecretString']

            except Exception as e:
                # Stack not found or insufficient permissions
                # This is expected in Workshop Studio where env vars are pre-set
                pass

        except ImportError:
            # boto3 not available
            pass

    # Validate we have all required credentials
    if not neo4j_uri:
        raise ValueError(
            "NEO4J_URI not found. Set NEO4J_URI environment variable or ensure "
            "you're running in a Workshop Studio environment."
        )

    if not neo4j_password:
        raise ValueError(
            "NEO4J_PASSWORD not found. Set NEO4J_PASSWORD environment variable or ensure "
            "you're running in a Workshop Studio environment."
        )

    return neo4j_uri, neo4j_user, neo4j_password


def print_neo4j_config():
    """Print Neo4j configuration (without exposing the full password)."""
    try:
        uri, user, password = get_neo4j_credentials()
        print(f"✓ Neo4j URI: {uri}")
        print(f"✓ Neo4j User: {user}")
        print(f"✓ Neo4j Password: {password[:4]}****")
    except ValueError as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # When run as a script, print configuration
    print_neo4j_config()
