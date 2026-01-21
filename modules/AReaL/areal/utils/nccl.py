import datetime

# For large models, generation may consume more than 3600s.
# We set a large value to avoid NCCL timeout issues during generaiton.
NCCL_DEFAULT_TIMEOUT = datetime.timedelta(seconds=7200)
