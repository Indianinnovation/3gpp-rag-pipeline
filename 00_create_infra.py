"""
00_create_infra.py
==================
Run ONCE to create every AWS resource the POC needs.
Uses RDS PostgreSQL with pgvector instead of OpenSearch Serverless.

Usage:
    python 00_create_infra.py

After this script finishes, copy the printed RDS endpoint into config.py.
"""

import boto3
import json
import time
from config import (
    AWS_REGION, S3_LANDING_BUCKET, S3_PROCESSED_BUCKET,
    DYNAMODB_TABLE, PG_DATABASE, PG_USER, EMBED_DIM
)

ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
RDS_INSTANCE_ID = "tgpp-rag-poc"
PG_PASSWORD = "Change_Me_123!"  # Change this!


# ─────────────────────────────────────────────────────────────────────────────
# 1. S3 buckets
# ─────────────────────────────────────────────────────────────────────────────
def create_s3_buckets():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    for bucket in [S3_LANDING_BUCKET, S3_PROCESSED_BUCKET]:
        try:
            if AWS_REGION == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
                )
            if bucket == S3_PROCESSED_BUCKET:
                s3.put_bucket_versioning(
                    Bucket=bucket, VersioningConfiguration={"Status": "Enabled"}
                )
            print(f"  ✓ S3 bucket created: {bucket}")
        except s3.exceptions.BucketAlreadyOwnedByYou:
            print(f"  · S3 bucket already exists: {bucket}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DynamoDB manifest table
# ─────────────────────────────────────────────────────────────────────────────
def create_dynamodb_table():
    ddb = boto3.client("dynamodb", region_name=AWS_REGION)
    try:
        ddb.create_table(
            TableName=DYNAMODB_TABLE,
            AttributeDefinitions=[{"AttributeName": "file_path", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "file_path", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
            Tags=[{"Key": "project", "Value": "3gpp-rag-poc"}]
        )
        ddb.get_waiter("table_exists").wait(TableName=DYNAMODB_TABLE)
        print(f"  ✓ DynamoDB table created: {DYNAMODB_TABLE}")
    except ddb.exceptions.ResourceInUseException:
        print(f"  · DynamoDB table already exists: {DYNAMODB_TABLE}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Security group for RDS
# ─────────────────────────────────────────────────────────────────────────────
def create_security_group() -> str:
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    sg_name = "3gpp-rag-poc-rds-sg"

    resp = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [sg_name]}]
    )
    if resp["SecurityGroups"]:
        sg_id = resp["SecurityGroups"][0]["GroupId"]
        print(f"  · Security group already exists: {sg_id}")
        return sg_id

    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    resp = ec2.create_security_group(
        GroupName=sg_name,
        Description="PostgreSQL access for 3GPP RAG POC",
        VpcId=vpc_id
    )
    sg_id = resp["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 5432, "ToPort": 5432,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "POC only"}]
        }]
    )
    print(f"  ✓ Security group created: {sg_id}")
    return sg_id


# ─────────────────────────────────────────────────────────────────────────────
# 4. RDS PostgreSQL instance
# ─────────────────────────────────────────────────────────────────────────────
def create_rds_instance(sg_id: str) -> str:
    rds = boto3.client("rds", region_name=AWS_REGION)

    try:
        rds.create_db_instance(
            DBInstanceIdentifier=RDS_INSTANCE_ID,
            DBInstanceClass="db.t3.micro",
            Engine="postgres",
            EngineVersion="16.4",
            MasterUsername=PG_USER,
            MasterUserPassword=PG_PASSWORD,
            DBName=PG_DATABASE,
            AllocatedStorage=20,
            StorageType="gp3",
            PubliclyAccessible=True,
            VpcSecurityGroupIds=[sg_id],
            BackupRetentionPeriod=0,
            Tags=[{"Key": "project", "Value": "3gpp-rag-poc"}]
        )
        print(f"  ✓ RDS instance creating: {RDS_INSTANCE_ID}")
    except rds.exceptions.DBInstanceAlreadyExistsFault:
        print(f"  · RDS instance already exists: {RDS_INSTANCE_ID}")

    print("  ⏳ Waiting for RDS to become available (5–10 min) …")
    waiter = rds.get_waiter("db_instance_available")
    waiter.wait(DBInstanceIdentifier=RDS_INSTANCE_ID, WaiterConfig={"Delay": 30, "MaxAttempts": 40})

    resp = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE_ID)
    endpoint = resp["DBInstances"][0]["Endpoint"]["Address"]
    print(f"  ✓ RDS AVAILABLE: {endpoint}")
    return endpoint


# ─────────────────────────────────────────────────────────────────────────────
# 5. Initialize pgvector extension + table
# ─────────────────────────────────────────────────────────────────────────────
def init_pgvector(host: str):
    import psycopg2
    conn = psycopg2.connect(host=host, port=5432, dbname=PG_DATABASE,
                            user=PG_USER, password=PG_PASSWORD)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT, spec_series TEXT,
            spec_number TEXT, release TEXT, section_path TEXT,
            doc_type TEXT DEFAULT 'TS', chunk_text TEXT, summary TEXT,
            keywords TEXT[], hyp_questions TEXT[], token_count INTEGER,
            source_s3_key TEXT, embedding vector({EMBED_DIM}),
            indexed_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
        USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_text_search_idx ON chunks
        USING gin (to_tsvector('english', chunk_text));
    """)
    print("  ✓ pgvector extension + chunks table + indexes created")
    cur.close()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  3GPP RAG POC — Infrastructure Setup (pgvector)")
    print("=" * 60)

    print("\n[1/5] S3 Buckets")
    create_s3_buckets()

    print("\n[2/5] DynamoDB Table")
    create_dynamodb_table()

    print("\n[3/5] Security Group")
    sg_id = create_security_group()

    print("\n[4/5] RDS PostgreSQL Instance")
    endpoint = create_rds_instance(sg_id)

    print("\n[5/5] Initialize pgvector")
    init_pgvector(endpoint)

    print("\n" + "=" * 60)
    print("  ✅ ALL DONE")
    print("=" * 60)
    print(f"\n  → Update config.py with:")
    print(f'    PG_HOST     = "{endpoint}"')
    print(f'    PG_PASSWORD = "{PG_PASSWORD}"')
    print(f"\n  → Then run: python 01_ftp_crawler.py")


if __name__ == "__main__":
    main()
