"""
cleanup.py
==========
Tears down all AWS resources created by the 3GPP RAG POC.
Run this when you're done to stop all charges.

Usage:
    python cleanup.py              # interactive confirmation
    python cleanup.py --force      # skip confirmation
"""

import argparse
import boto3
from config import (
    AWS_REGION, S3_LANDING_BUCKET, S3_PROCESSED_BUCKET, DYNAMODB_TABLE
)

RDS_INSTANCE_ID = "tgpp-rag-poc"
SG_NAME = "3gpp-rag-poc-rds-sg"


def delete_s3_bucket(bucket_name: str):
    """Empty and delete an S3 bucket."""
    s3 = boto3.resource("s3", region_name=AWS_REGION)
    try:
        bucket = s3.Bucket(bucket_name)
        bucket.object_versions.all().delete()
        bucket.objects.all().delete()
        bucket.delete()
        print(f"  ✓ Deleted S3 bucket: {bucket_name}")
    except Exception as e:
        print(f"  · S3 bucket {bucket_name}: {e}")


def delete_dynamodb_table():
    """Delete the DynamoDB manifest table."""
    ddb = boto3.client("dynamodb", region_name=AWS_REGION)
    try:
        ddb.delete_table(TableName=DYNAMODB_TABLE)
        print(f"  ✓ Deleted DynamoDB table: {DYNAMODB_TABLE}")
    except Exception as e:
        print(f"  · DynamoDB: {e}")


def delete_rds_instance():
    """Delete the RDS PostgreSQL instance (skip final snapshot)."""
    rds = boto3.client("rds", region_name=AWS_REGION)
    try:
        rds.delete_db_instance(
            DBInstanceIdentifier=RDS_INSTANCE_ID,
            SkipFinalSnapshot=True,
            DeleteAutomatedBackups=True
        )
        print(f"  ✓ Deleting RDS instance: {RDS_INSTANCE_ID} (takes 2-5 min)")
    except Exception as e:
        print(f"  · RDS: {e}")


def delete_security_group():
    """Delete the RDS security group."""
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    try:
        resp = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
        )
        if resp["SecurityGroups"]:
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            ec2.delete_security_group(GroupId=sg_id)
            print(f"  ✓ Deleted security group: {sg_id}")
        else:
            print(f"  · Security group not found")
    except Exception as e:
        print(f"  · Security group: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  3GPP RAG POC — Resource Cleanup")
    print("=" * 60)
    print("\nThis will DELETE all resources:")
    print(f"  - S3: {S3_LANDING_BUCKET}, {S3_PROCESSED_BUCKET}")
    print(f"  - DynamoDB: {DYNAMODB_TABLE}")
    print(f"  - RDS: {RDS_INSTANCE_ID}")
    print(f"  - Security Group: {SG_NAME}")

    if not args.force:
        confirm = input("\nType 'DELETE' to confirm: ")
        if confirm != "DELETE":
            print("Aborted.")
            return

    print("\n[1/5] Deleting RDS instance …")
    delete_rds_instance()

    print("\n[2/5] Deleting S3 buckets …")
    delete_s3_bucket(S3_LANDING_BUCKET)
    delete_s3_bucket(S3_PROCESSED_BUCKET)

    print("\n[3/5] Deleting DynamoDB table …")
    delete_dynamodb_table()

    print("\n[4/5] Waiting for RDS deletion before removing security group …")
    print("  (Run 'python cleanup.py --force' again in 5 min to delete the SG)")

    print("\n" + "=" * 60)
    print("  ✅ Cleanup initiated — charges will stop within minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
