import getpass
import os


def get_user_tmp():
    user_tmp = os.path.join(os.path.expanduser("~"), ".cache", "realhf")
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp
