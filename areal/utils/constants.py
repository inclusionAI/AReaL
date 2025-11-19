import datetime

# For large models, generation may consume more than 7200s.
# We set a large value to avoid timeout issues during generation.
DIST_GROUP_DEFAULT_TIMEOUT = datetime.timedelta(seconds=7200)
