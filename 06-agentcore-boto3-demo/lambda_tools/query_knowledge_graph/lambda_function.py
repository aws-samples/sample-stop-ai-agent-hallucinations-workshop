# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda: query_knowledge_graph — runs read-only Cypher queries against Neo4j.

This Lambda runs in the workshop VPC with the Neo4j security group and reaches
Neo4j on port 7687 (at AWS events, the team's internal Network Load Balancer).

Environment variables:
  NEO4J_HOST: Neo4j host name or private IP
  NEO4J_PASSWORD_SECRET_ARN: Secrets Manager ARN for the Neo4j password
"""

import json
import os
import re

import boto3
from neo4j import GraphDatabase, RoutingControl

secrets = boto3.client("secretsmanager")

NEO4J_HOST = os.environ["NEO4J_HOST"]
NEO4J_PASSWORD_SECRET_ARN = os.environ["NEO4J_PASSWORD_SECRET_ARN"]

_driver = None

# The agent writes the Cypher, so only read queries are accepted. Read routing alone
# is not access control (Neo4j driver manual), so write clauses are also rejected.
WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|CALL)\b", re.IGNORECASE
)


def _get_driver():
    global _driver
    if _driver is None:
        secret_string = secrets.get_secret_value(SecretId=NEO4J_PASSWORD_SECRET_ARN)["SecretString"]
        # The secret is either a JSON document with a "password" field or the
        # password itself in plain text. Both are accepted.
        try:
            password = json.loads(secret_string).get("password", secret_string)
        except (json.JSONDecodeError, AttributeError):
            password = secret_string
        uri = f"bolt://{NEO4J_HOST}:7687"
        _driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    return _driver


def handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    cypher_query = body.get("cypher_query", "")

    if not cypher_query:
        return {"statusCode": 400, "body": "ERROR: cypher_query is required."}
    if WRITE_CLAUSES.search(cypher_query):
        return {"statusCode": 400, "body": "ERROR: only read-only Cypher (MATCH ... RETURN) is allowed."}

    try:
        records, _, _ = _get_driver().execute_query(cypher_query, routing_=RoutingControl.READ)
        records = [dict(record) for record in records][:15]

        if not records:
            return {"statusCode": 200, "body": "No results found."}

        formatted = json.dumps(records, indent=2, default=str)
        return {"statusCode": 200, "body": f"Query returned {len(records)} result(s):\n{formatted}"}

    except Exception as e:
        return {"statusCode": 200, "body": f"Query error: {str(e)}"}
