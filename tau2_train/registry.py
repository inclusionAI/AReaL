
ENV_DIC = {}

from tau2_train.domains.airline.environment import get_environment as get_airline_env
from tau2_train.domains.airline.environment import get_tasks as get_airline_tasks

from tau2_train.domains.retail.environment import get_environment as get_retail_env
from tau2_train.domains.retail.environment import get_tasks as get_retail_tasks


from tau2_train.domains.telecom.environment import get_environment as get_telecom_env
from tau2_train.domains.telecom.environment import get_tasks as get_telecom_tasks

ENV_DIC = {
    "airline": get_airline_env,
    "retail": get_retail_env,
    "telecom": get_telecom_env,
}

TASK_DIC = {
    "airline": get_airline_tasks,
    "retail": get_retail_tasks,
    "telecom": get_telecom_tasks,
}