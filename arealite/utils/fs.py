import getpass
import os


def get_user_tmp():
    user = getpass.getuser()
    user_tmp = os.path.join("/home", user, ".cache", "realhf")
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp
