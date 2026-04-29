# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: query_knowledge_graph — executes Cypher queries against Neo4j.

Neo4j runs on the Code Editor EC2 instance. This Lambda runs in the same
VPC/Security Group to access Neo4j via the EC2's private IP on port 7687.

Environment variables:
  NEO4J_HOST: Private IP of the EC2 instance
  NEO4J_PASSWORD_SECRET_ARN: Secrets Manager ARN for the Neo4j password
"""

import json
import os

import boto3
from neo4j import GraphDatabase

secrets = boto3.client("secretsmanager")

NEO4J_HOST = os.environ["NEO4J_HOST"]
NEO4J_PASSWORD_SECRET_ARN = os.environ["NEO4J_PASSWORD_SECRET_ARN"]

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        password = secrets.get_secret_value(SecretId=NEO4J_PASSWORD_SECRET_ARN)["SecretString"]
        uri = f"bolt://{NEO4J_HOST}:7687"
        _driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    return _driver


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    cypher_query = body.get("cypher_query", "")

    if not cypher_query:
        return {"statusCode": 400, "body": "ERROR: cypher_query is required."}

    try:
        driver = _get_driver()
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [dict(record) for record in result][:15]

        if not records:
            return {"statusCode": 200, "body": "No results found."}

        formatted = json.dumps(records, indent=2, default=str)
        return {"statusCode": 200, "body": f"Query returned {len(records)} result(s):\n{formatted}"}

    except Exception as e:
        return {"statusCode": 200, "body": f"Query error: {str(e)}"}
