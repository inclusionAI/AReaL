#!/usr/bin/env python3
"""
Utility script to upload test logs to cloud services or email them.

Supports:
- Email (SMTP)
- Google Drive (via rclone)
- AWS S3
- Hugging Face Hub
- W&B Artifacts
- Generic webhook/API endpoint

Usage:
    python upload_logs.py --log-dir /path/to/logs --method email --email-to user@example.com
    python upload_logs.py --log-dir /path/to/logs --method gdrive --gdrive-folder-id FOLDER_ID
    python upload_logs.py --log-dir /path/to/logs --method s3 --s3-bucket my-bucket
"""

import argparse
import os
import sys
import glob
import smtplib
import subprocess
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import List, Optional
import tarfile
import tempfile


def upload_via_email(
    log_files: List[str],
    email_to: str,
    email_from: Optional[str] = None,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
    subject: str = "Training Test Logs",
):
    """Upload logs via email."""
    print(f"📧 Uploading {len(log_files)} log files via email to {email_to}...")
    
    # Get email credentials from environment if not provided
    email_from = email_from or os.environ.get("EMAIL_FROM")
    smtp_user = smtp_user or os.environ.get("SMTP_USER") or email_from
    smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD")
    
    if not email_from or not smtp_password:
        print("❌ Error: Email credentials not provided. Set EMAIL_FROM and SMTP_PASSWORD environment variables.")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = subject
        
        # Create body
        body = f"""
Training test logs attached.

Files:
{chr(10).join(f'  - {os.path.basename(f)}' for f in log_files)}

Total files: {len(log_files)}
"""
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach files
        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(log_file)}'
                    )
                    msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(email_from, email_to, text)
        server.quit()
        
        print(f"✅ Successfully sent {len(log_files)} log files via email!")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False


def upload_via_gdrive(
    log_files: List[str],
    folder_id: Optional[str] = None,
    rclone_config: Optional[str] = None,
):
    """Upload logs to Google Drive using rclone."""
    print(f"☁️  Uploading {len(log_files)} log files to Google Drive...")
    
    # Check if rclone is installed
    try:
        subprocess.run(["rclone", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: rclone is not installed. Install it with: curl https://rclone.org/install.sh | sudo bash")
        return False
    
    # Create a tar archive of all logs
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        tar_path = tmp_file.name
    
    try:
        with tarfile.open(tar_path, 'w:gz') as tar:
            for log_file in log_files:
                if os.path.exists(log_file):
                    tar.add(log_file, arcname=os.path.basename(log_file))
        
        # Upload using rclone
        remote_name = rclone_config or os.environ.get("RCLONE_REMOTE", "gdrive")
        remote_path = f"{remote_name}:areal-training-logs"
        if folder_id:
            remote_path = f"{remote_name}:{folder_id}"
        
        cmd = ["rclone", "copy", tar_path, remote_path, "--progress"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully uploaded logs to Google Drive: {remote_path}")
            return True
        else:
            print(f"❌ Error uploading to Google Drive: {result.stderr}")
            return False
    finally:
        if os.path.exists(tar_path):
            os.remove(tar_path)


def upload_via_s3(
    log_files: List[str],
    bucket_name: str,
    s3_prefix: str = "areal-training-logs",
    aws_access_key: Optional[str] = None,
    aws_secret_key: Optional[str] = None,
):
    """Upload logs to AWS S3."""
    print(f"☁️  Uploading {len(log_files)} log files to S3 bucket: {bucket_name}...")
    
    try:
        import boto3
    except ImportError:
        print("❌ Error: boto3 is not installed. Install it with: pip install boto3")
        return False
    
    # Get credentials from environment if not provided
    aws_access_key = aws_access_key or os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = aws_secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key or not aws_secret_key:
        print("❌ Error: AWS credentials not provided. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        return False
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        for log_file in log_files:
            if os.path.exists(log_file):
                s3_key = f"{s3_prefix}/{os.path.basename(log_file)}"
                s3_client.upload_file(log_file, bucket_name, s3_key)
                print(f"  ✅ Uploaded: {os.path.basename(log_file)}")
        
        print(f"✅ Successfully uploaded {len(log_files)} log files to S3!")
        return True
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")
        return False


def upload_via_hf_hub(
    log_files: List[str],
    repo_id: str,
    hf_token: Optional[str] = None,
):
    """Upload logs to Hugging Face Hub."""
    print(f"🤗 Uploading {len(log_files)} log files to Hugging Face Hub: {repo_id}...")
    
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Error: huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        return False
    
    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        print("❌ Error: Hugging Face token not provided. Set HF_TOKEN environment variable.")
        return False
    
    try:
        api = HfApi(token=hf_token)
        
        for log_file in log_files:
            if os.path.exists(log_file):
                api.upload_file(
                    path_or_fileobj=log_file,
                    path_in_repo=f"test_logs/{os.path.basename(log_file)}",
                    repo_id=repo_id,
                    repo_type="dataset",  # Use dataset type for logs
                )
                print(f"  ✅ Uploaded: {os.path.basename(log_file)}")
        
        print(f"✅ Successfully uploaded {len(log_files)} log files to Hugging Face Hub!")
        return True
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face Hub: {e}")
        return False


def upload_via_wandb_artifact(
    log_files: List[str],
    project: str = "gsm8k-grpo-cloud",
    run_name: Optional[str] = None,
):
    """Upload logs as W&B artifacts."""
    print(f"📊 Uploading {len(log_files)} log files as W&B artifacts...")
    
    try:
        import wandb
    except ImportError:
        print("❌ Error: wandb is not installed. Install it with: pip install wandb")
        return False
    
    try:
        # Initialize or resume W&B run
        wandb.init(project=project, name=run_name, resume="allow")
        
        # Create artifact
        artifact = wandb.Artifact("test_logs", type="logs")
        
        for log_file in log_files:
            if os.path.exists(log_file):
                artifact.add_file(log_file)
        
        wandb.log_artifact(artifact)
        print(f"✅ Successfully uploaded {len(log_files)} log files as W&B artifacts!")
        return True
    except Exception as e:
        print(f"❌ Error uploading to W&B: {e}")
        return False


def upload_via_webhook(
    log_files: List[str],
    webhook_url: str,
    api_key: Optional[str] = None,
):
    """Upload logs via webhook/API endpoint."""
    print(f"🌐 Uploading {len(log_files)} log files via webhook...")
    
    try:
        import requests
    except ImportError:
        print("❌ Error: requests is not installed. Install it with: pip install requests")
        return False
    
    try:
        # Create a tar archive
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tar_path = tmp_file.name
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            for log_file in log_files:
                if os.path.exists(log_file):
                    tar.add(log_file, arcname=os.path.basename(log_file))
        
        # Upload via POST request
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        with open(tar_path, 'rb') as f:
            files = {'file': (os.path.basename(tar_path), f, 'application/gzip')}
            response = requests.post(webhook_url, files=files, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Successfully uploaded logs via webhook!")
            return True
        else:
            print(f"❌ Error uploading via webhook: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error uploading via webhook: {e}")
        return False
    finally:
        if os.path.exists(tar_path):
            os.remove(tar_path)


def normalize_method(value):
    """Normalize method name to lowercase and validate."""
    value = value.lower()
    valid_methods = ["email", "gdrive", "s3", "hf", "wandb", "webhook"]
    if value not in valid_methods:
        raise argparse.ArgumentTypeError(
            f"Invalid method '{value}'. Choose from: {', '.join(valid_methods)}"
        )
    return value


def main():
    parser = argparse.ArgumentParser(description="Upload test logs to cloud services or email")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing log files",
    )
    parser.add_argument(
        "--method",
        type=normalize_method,
        required=True,
        help="Upload method (case-insensitive: email, gdrive, s3, hf, wandb, webhook)",
    )
    
    # Email options
    parser.add_argument("--email-to", type=str, help="Email recipient")
    parser.add_argument("--email-from", type=str, help="Email sender (or set EMAIL_FROM env var)")
    parser.add_argument("--smtp-server", type=str, default="smtp.gmail.com", help="SMTP server")
    parser.add_argument("--smtp-port", type=int, default=587, help="SMTP port")
    parser.add_argument("--smtp-user", type=str, help="SMTP username (or set SMTP_USER env var)")
    parser.add_argument("--smtp-password", type=str, help="SMTP password (or set SMTP_PASSWORD env var)")
    parser.add_argument("--email-subject", type=str, default="Training Test Logs", help="Email subject")
    
    # Google Drive options
    parser.add_argument("--gdrive-folder-id", type=str, help="Google Drive folder ID")
    parser.add_argument("--rclone-remote", type=str, help="Rclone remote name (or set RCLONE_REMOTE env var)")
    
    # S3 options
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, default="areal-training-logs", help="S3 key prefix")
    parser.add_argument("--aws-access-key", type=str, help="AWS access key (or set AWS_ACCESS_KEY_ID env var)")
    parser.add_argument("--aws-secret-key", type=str, help="AWS secret key (or set AWS_SECRET_ACCESS_KEY env var)")
    
    # Hugging Face options
    parser.add_argument("--hf-repo-id", type=str, help="Hugging Face repository ID (e.g., username/dataset-name)")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token (or set HF_TOKEN env var)")
    
    # W&B options
    parser.add_argument("--wandb-project", type=str, default="gsm8k-grpo-cloud", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")
    
    # Webhook options
    parser.add_argument("--webhook-url", type=str, help="Webhook URL")
    parser.add_argument("--webhook-api-key", type=str, help="Webhook API key")
    
    # Filter options
    parser.add_argument("--pattern", type=str, default="*.log", help="File pattern to match (default: *.log)")
    parser.add_argument("--latest-only", action="store_true", help="Only upload the latest log files (baseline and trained)")
    
    args = parser.parse_args()
    
    # Find log files
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ Error: Log directory does not exist: {log_dir}")
        return 1
    
    if args.latest_only:
        # Find latest baseline and trained logs
        baseline_logs = sorted(log_dir.glob("test_model_baseline_*.log"), key=os.path.getmtime, reverse=True)
        trained_logs = sorted(log_dir.glob("test_model_trained_*.log"), key=os.path.getmtime, reverse=True)
        log_files = []
        if baseline_logs:
            log_files.append(str(baseline_logs[0]))
        if trained_logs:
            log_files.append(str(trained_logs[0]))
    else:
        log_files = [str(f) for f in log_dir.glob(args.pattern)]
    
    if not log_files:
        print(f"⚠️  No log files found in {log_dir} matching pattern {args.pattern}")
        return 0
    
    print(f"Found {len(log_files)} log file(s) to upload:")
    for f in log_files:
        print(f"  - {os.path.basename(f)}")
    
    # Upload based on method
    success = False
    if args.method == "email":
        if not args.email_to:
            print("❌ Error: --email-to is required for email method")
            return 1
        success = upload_via_email(
            log_files,
            args.email_to,
            args.email_from,
            args.smtp_server,
            args.smtp_port,
            args.smtp_user,
            args.smtp_password,
            args.email_subject,
        )
    elif args.method == "gdrive":
        success = upload_via_gdrive(log_files, args.gdrive_folder_id, args.rclone_remote)
    elif args.method == "s3":
        if not args.s3_bucket:
            print("❌ Error: --s3-bucket is required for S3 method")
            return 1
        success = upload_via_s3(
            log_files,
            args.s3_bucket,
            args.s3_prefix,
            args.aws_access_key,
            args.aws_secret_key,
        )
    elif args.method == "hf":
        if not args.hf_repo_id:
            print("❌ Error: --hf-repo-id is required for Hugging Face method")
            return 1
        success = upload_via_hf_hub(log_files, args.hf_repo_id, args.hf_token)
    elif args.method == "wandb":
        success = upload_via_wandb_artifact(log_files, args.wandb_project, args.wandb_run_name)
    elif args.method == "webhook":
        if not args.webhook_url:
            print("❌ Error: --webhook-url is required for webhook method")
            return 1
        success = upload_via_webhook(log_files, args.webhook_url, args.webhook_api_key)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

