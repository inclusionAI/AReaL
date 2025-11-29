# Auto-Upload Test Logs Guide

This guide explains how to automatically upload test logs to cloud services or email them after training completes.

## Overview

After training and testing complete, the system can automatically upload the latest test logs (baseline and trained model results) to:
- 📧 **Email** (SMTP)
- ☁️ **Google Drive** (via rclone)
- ☁️ **AWS S3**
- 🤗 **Hugging Face Hub**
- 📊 **Weights & Biases** (as artifacts)
- 🌐 **Generic webhook/API endpoint**

## Quick Start

### Email (Easiest)

Set these environment variables in your RunPod pod:

```bash
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=your-email@example.com
export EMAIL_FROM=your-sender@example.com
export SMTP_PASSWORD=your-app-password  # Gmail: use App Password, not regular password
```

**For Gmail:**
1. Enable 2-factor authentication
2. Generate an App Password: https://myaccount.google.com/apppasswords
3. Use the App Password as `SMTP_PASSWORD`

### Google Drive

1. **Install rclone** (if not already installed):
   ```bash
   curl https://rclone.org/install.sh | sudo bash
   ```

2. **Configure rclone**:
   ```bash
   rclone config
   # Follow prompts to set up Google Drive
   # Name it "gdrive" (or set RCLONE_REMOTE env var)
   ```

3. **Set environment variables**:
   ```bash
   export AUTO_UPLOAD_LOGS_METHOD=gdrive
   export AUTO_UPLOAD_GDRIVE_FOLDER_ID=YOUR_FOLDER_ID  # Optional: specific folder
   export RCLONE_REMOTE=gdrive  # Optional: if you named it differently
   ```

### AWS S3

```bash
export AUTO_UPLOAD_LOGS_METHOD=s3
export AUTO_UPLOAD_S3_BUCKET=my-bucket-name
export AUTO_UPLOAD_S3_PREFIX=areal-training-logs  # Optional: folder path
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

### Hugging Face Hub

```bash
export AUTO_UPLOAD_LOGS_METHOD=hf
export AUTO_UPLOAD_HF_REPO_ID=username/dataset-name
export HF_TOKEN=your-hf-token
```

### Weights & Biases

```bash
export AUTO_UPLOAD_LOGS_METHOD=wandb
export AUTO_UPLOAD_WANDB_PROJECT=gsm8k-grpo-cloud  # Optional
export AUTO_UPLOAD_WANDB_RUN_NAME=my-run-name  # Optional
export WANDB_API_KEY=your-wandb-key  # Already set for training
```

### Webhook/API Endpoint

```bash
export AUTO_UPLOAD_LOGS_METHOD=webhook
export AUTO_UPLOAD_WEBHOOK_URL=https://your-api.com/upload
export AUTO_UPLOAD_WEBHOOK_API_KEY=your-api-key  # Optional
```

## Setting Environment Variables in RunPod

### Method 1: In RunPod Docker Command

Add environment variables to your RunPod startup command:

```bash
bash -c "export AUTO_UPLOAD_LOGS_METHOD=email && export AUTO_UPLOAD_EMAIL_TO=your@email.com && export EMAIL_FROM=sender@email.com && export SMTP_PASSWORD=your-password && ... (rest of your command)"
```

### Method 2: In RunPod Template Environment Variables

1. Go to RunPod Templates: https://www.runpod.io/console/templates
2. Edit your template
3. Add environment variables in the "Environment Variables" section:
   - `AUTO_UPLOAD_LOGS_METHOD=email`
   - `AUTO_UPLOAD_EMAIL_TO=your@email.com`
   - `EMAIL_FROM=sender@email.com`
   - `SMTP_PASSWORD=your-password`

### Method 3: In the Shell Script (Temporary)

Edit `examples/cloud_gsm8k/run_training_cloud.sh` and add at the top:

```bash
# Auto-upload configuration
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=your@email.com
export EMAIL_FROM=sender@email.com
export SMTP_PASSWORD=your-password
```

## Manual Upload

You can also manually upload logs after training completes:

```bash
# Upload latest logs via email
python3 examples/cloud_gsm8k/upload_logs.py \
    --log-dir /workspace/outputs/grpo/test_logs \
    --method email \
    --email-to your@email.com \
    --latest-only

# Upload all logs to Google Drive
python3 examples/cloud_gsm8k/upload_logs.py \
    --log-dir /workspace/outputs/grpo/test_logs \
    --method gdrive \
    --gdrive-folder-id YOUR_FOLDER_ID

# Upload to S3
python3 examples/cloud_gsm8k/upload_logs.py \
    --log-dir /workspace/outputs/grpo/test_logs \
    --method s3 \
    --s3-bucket my-bucket \
    --s3-prefix areal-logs
```

## What Gets Uploaded

By default, only the **latest** baseline and trained model test logs are uploaded (using `--latest-only`). This includes:
- `test_model_baseline_YYYYMMDD_HHMMSS.log` (most recent)
- `test_model_trained_YYYYMMDD_HHMMSS.log` (most recent)

To upload all logs, remove the `--latest-only` flag or set it to `false`.

## Email Configuration Examples

### Gmail

```bash
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=recipient@gmail.com
export EMAIL_FROM=sender@gmail.com
export SMTP_PASSWORD=xxxx-xxxx-xxxx-xxxx  # App Password
# SMTP server defaults to smtp.gmail.com:587
```

### Outlook/Office 365

```bash
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=recipient@outlook.com
export EMAIL_FROM=sender@outlook.com
export SMTP_PASSWORD=your-password
export SMTP_SERVER=smtp-mail.outlook.com
export SMTP_PORT=587
```

### Custom SMTP Server

```bash
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=recipient@example.com
export EMAIL_FROM=sender@example.com
export SMTP_SERVER=smtp.example.com
export SMTP_PORT=587
export SMTP_USER=sender@example.com
export SMTP_PASSWORD=your-password
```

## Troubleshooting

### Email Not Sending

1. **Check credentials**: Verify `EMAIL_FROM` and `SMTP_PASSWORD` are correct
2. **Gmail**: Must use App Password, not regular password
3. **Check SMTP settings**: Some providers use different ports (465 for SSL, 587 for TLS)
4. **Firewall**: Ensure RunPod can access SMTP servers

### Google Drive Upload Fails

1. **Install rclone**: `curl https://rclone.org/install.sh | sudo bash`
2. **Configure rclone**: Run `rclone config` to set up Google Drive
3. **Check permissions**: Ensure rclone has access to the folder

### S3 Upload Fails

1. **Check credentials**: Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
2. **Check bucket permissions**: Ensure IAM user has `s3:PutObject` permission
3. **Check bucket exists**: Verify bucket name is correct

### Hugging Face Upload Fails

1. **Check token**: Verify `HF_TOKEN` is valid
2. **Check repo exists**: Repository must exist (create it on HF Hub first)
3. **Check permissions**: Ensure token has write access to the repo

### W&B Upload Fails

1. **Check W&B is initialized**: Training must have initialized W&B first
2. **Check project name**: Verify project exists or will be created
3. **Check API key**: Verify `WANDB_API_KEY` is set

## Security Notes

⚠️ **Important**: Never commit credentials to git!

- Use RunPod environment variables or secrets management
- For email: Use App Passwords (Gmail) instead of regular passwords
- For S3: Use IAM users with minimal permissions
- For HF/W&B: Use tokens with appropriate scopes

## Examples

### Complete Email Setup (Gmail)

```bash
# In RunPod Docker command or template:
export AUTO_UPLOAD_LOGS_METHOD=email
export AUTO_UPLOAD_EMAIL_TO=your-email@gmail.com
export EMAIL_FROM=your-sender@gmail.com
export SMTP_PASSWORD=xxxx-xxxx-xxxx-xxxx  # Gmail App Password
```

### Complete Google Drive Setup

```bash
# In RunPod pod terminal:
curl https://rclone.org/install.sh | sudo bash
rclone config  # Follow prompts, name it "gdrive"

# In RunPod Docker command or template:
export AUTO_UPLOAD_LOGS_METHOD=gdrive
export RCLONE_REMOTE=gdrive
```

### Complete S3 Setup

```bash
# In RunPod Docker command or template:
export AUTO_UPLOAD_LOGS_METHOD=s3
export AUTO_UPLOAD_S3_BUCKET=my-areal-logs
export AUTO_UPLOAD_S3_PREFIX=test-logs
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
```

## Summary

1. ✅ Set `AUTO_UPLOAD_LOGS_METHOD` to your preferred method
2. ✅ Set method-specific environment variables
3. ✅ Run training - logs will auto-upload after tests complete
4. ✅ Check your email/cloud storage for the logs!

The upload happens automatically after tests complete, so you don't need to manually download logs from RunPod!

