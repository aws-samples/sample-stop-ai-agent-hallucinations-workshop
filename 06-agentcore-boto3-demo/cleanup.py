# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Cleanup ALL resources created by the AgentCore workshop notebook.
Standalone — no imports from other files.

Usage: python3 cleanup.py
"""
import os
import select

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

dynamodb = boto3.client("dynamodb", region_name=REGION)
iam = boto3.client("iam")
lambda_client = boto3.client("lambda", region_name=REGION)
agentcore = boto3.client("bedrock-agentcore-control", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
codebuild = boto3.client("codebuild", region_name=REGION)
ecr = boto3.client("ecr", region_name=REGION)

ECR_REPO_NAME = "bedrock-agentcore-hotelbookingagent"
CODEBUILD_PROJECT = f"{ECR_REPO_NAME}-builder"
CODEBUILD_ROLE = f"AmazonBedrockAgentCoreSDKCodeBuild-{REGION}-{ACCOUNT_ID[:8]}"

RUNTIME_NAME = "HotelBookingAgent"
GATEWAY_NAME = "HotelBookingGateway"
LAMBDA_TOOLS = [
    "search_available_hotels", "book_hotel", "get_booking",
    "process_payment", "confirm_booking", "cancel_booking",
    "validate_booking_rules",
]
TABLE_NAMES = ["workshop-Hotels", "workshop-Bookings", "workshop-SteeringRules"]
ROLE_NAMES = ["workshop-LambdaExecutionRole", "workshop-AgentCoreExecutionRole"]
BUCKET_NAME = f"workshop-agent-code-{ACCOUNT_ID}-{REGION}"


def cleanup():
    print("=" * 60)
    print("CLEANUP: Removing ALL workshop resources")
    print("=" * 60)

    # 1. Delete AgentCore Runtime
    print("\n1. Deleting AgentCore Runtimes...")
    try:
        runtimes = agentcore.list_agent_runtimes().get("agentRuntimes", [])
        for rt in runtimes:
            if rt.get("agentRuntimeName") == RUNTIME_NAME:
                rid = rt["agentRuntimeId"]
                try:
                    agentcore.delete_agent_runtime(agentRuntimeId=rid)
                    print(f"  Deleting runtime: {rid}...")
                    # Wait for deletion
                    for _ in range(30):
                        try:
                            status = agentcore.get_agent_runtime(agentRuntimeId=rid).get("status")
                            if "DELET" in str(status):
                                select.select([], [], [], 5)
                            else:
                                break
                        except Exception:
                            break
                    print(f"  Deleted runtime: {rid}")
                except Exception as e:
                    print(f"  Error deleting runtime {rid}: {e}")
    except Exception as e:
        print(f"  Skip: {e}")

    # 2. Delete Gateway Targets and Gateway
    print("\n2. Deleting AgentCore Gateway + Targets...")
    try:
        gateways = agentcore.list_gateways().get("items", [])
        for gw in gateways:
            if gw.get("name") == GATEWAY_NAME:
                gw_id = gw["gatewayId"]
                # Delete all targets first
                try:
                    targets = agentcore.list_gateway_targets(gatewayIdentifier=gw_id).get("items", [])
                    for t in targets:
                        tid = t.get("targetId", t.get("id", ""))
                        tname = t.get("name", tid)
                        try:
                            agentcore.delete_gateway_target(gatewayIdentifier=gw_id, targetId=tid)
                            print(f"  Deleted target: {tname}")
                        except Exception as e:
                            print(f"  Error deleting target {tname}: {e}")
                except Exception as e:
                    print(f"  Error listing targets: {e}")

                # Wait for targets to be deleted
                select.select([], [], [], 5)

                # Delete gateway
                try:
                    agentcore.delete_gateway(gatewayIdentifier=gw_id)
                    print(f"  Deleting gateway: {gw_id}...")
                    for _ in range(30):
                        try:
                            status = agentcore.get_gateway(gatewayIdentifier=gw_id).get("status")
                            if "DELET" in str(status):
                                select.select([], [], [], 5)
                            else:
                                break
                        except Exception:
                            break
                    print(f"  Deleted gateway: {gw_id}")
                except Exception as e:
                    print(f"  Error deleting gateway {gw_id}: {e}")
    except Exception as e:
        print(f"  Skip: {e}")

    # 3. Delete Lambda functions
    print("\n3. Deleting Lambda functions...")
    for tool_name in LAMBDA_TOOLS:
        function_name = f"hotel-booking-{tool_name}"
        try:
            lambda_client.delete_function(FunctionName=function_name)
            print(f"  Deleted {function_name}")
        except lambda_client.exceptions.ResourceNotFoundException:
            pass
        except Exception as e:
            print(f"  Error: {function_name}: {e}")

    # 4. Delete IAM roles
    print("\n4. Deleting IAM roles...")
    for role_name in ROLE_NAMES:
        try:
            # Detach managed policies
            attached = iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]
            for pol in attached:
                iam.detach_role_policy(RoleName=role_name, PolicyArn=pol["PolicyArn"])
            # Delete inline policies
            inline = iam.list_role_policies(RoleName=role_name)["PolicyNames"]
            for pol_name in inline:
                iam.delete_role_policy(RoleName=role_name, PolicyName=pol_name)
            # Delete role
            iam.delete_role(RoleName=role_name)
            print(f"  Deleted {role_name}")
        except iam.exceptions.NoSuchEntityException:
            pass
        except Exception as e:
            print(f"  Error: {role_name}: {e}")

    # 5. Delete DynamoDB tables
    print("\n5. Deleting DynamoDB tables...")
    for table_name in TABLE_NAMES:
        try:
            dynamodb.delete_table(TableName=table_name)
            print(f"  Deleted {table_name}")
        except dynamodb.exceptions.ResourceNotFoundException:
            pass
        except Exception as e:
            print(f"  Error: {table_name}: {e}")

    # 6. Delete S3 bucket
    print("\n6. Deleting S3 bucket...")
    try:
        # Delete all objects first
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_NAME):
            for obj in page.get("Contents", []):
                s3.delete_object(Bucket=BUCKET_NAME, Key=obj["Key"])
        # Delete versions if versioned
        try:
            versions = s3.list_object_versions(Bucket=BUCKET_NAME)
            for v in versions.get("Versions", []):
                s3.delete_object(Bucket=BUCKET_NAME, Key=v["Key"], VersionId=v["VersionId"])
            for dm in versions.get("DeleteMarkers", []):
                s3.delete_object(Bucket=BUCKET_NAME, Key=dm["Key"], VersionId=dm["VersionId"])
        except Exception:
            print("  Note: bucket versioning not configured")
        s3.delete_bucket(Bucket=BUCKET_NAME)
        print(f"  Deleted {BUCKET_NAME}")
    except s3.exceptions.NoSuchBucket:
        pass
    except Exception as e:
        print(f"  Error: {e}")

    # 7. Delete CodeBuild project
    print("\n7. Deleting CodeBuild project...")
    try:
        codebuild.delete_project(name=CODEBUILD_PROJECT)
        print(f"  Deleted {CODEBUILD_PROJECT}")
    except Exception as e:
        if "does not exist" in str(e) or "not found" in str(e).lower():
            pass
        else:
            print(f"  Error: {e}")

    # 8. Delete ECR repository (and all images)
    print("\n8. Deleting ECR repository...")
    try:
        ecr.delete_repository(repositoryName=ECR_REPO_NAME, force=True)
        print(f"  Deleted {ECR_REPO_NAME}")
    except ecr.exceptions.RepositoryNotFoundException:
        pass
    except Exception as e:
        print(f"  Error: {e}")

    # 9. Delete CodeBuild IAM role
    print("\n9. Deleting CodeBuild IAM role...")
    try:
        cb_roles = iam.list_roles(PathPrefix="/")["Roles"]
        for role in cb_roles:
            if role["RoleName"].startswith("AmazonBedrockAgentCoreSDKCodeBuild"):
                rn = role["RoleName"]
                try:
                    for pol in iam.list_attached_role_policies(RoleName=rn)["AttachedPolicies"]:
                        iam.detach_role_policy(RoleName=rn, PolicyArn=pol["PolicyArn"])
                    for pn in iam.list_role_policies(RoleName=rn)["PolicyNames"]:
                        iam.delete_role_policy(RoleName=rn, PolicyName=pn)
                    iam.delete_role(RoleName=rn)
                    print(f"  Deleted {rn}")
                except Exception as e:
                    print(f"  Error: {rn}: {e}")
    except Exception as e:
        print(f"  Skip: {e}")

    # 10. Delete starter toolkit config (contains old runtime IDs)
    print("\n10. Deleting starter toolkit config...")
    import glob
    for config_file in glob.glob(".bedrock_agentcore*.yaml") + glob.glob(os.path.expanduser("~/.bedrock_agentcore*.yaml")):
        try:
            os.remove(config_file)
            print(f"  Deleted {config_file}")
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    cleanup()
